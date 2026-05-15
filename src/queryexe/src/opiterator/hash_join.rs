use super::OpIterator;
use crate::Managers;

use common::bytecode_expr::ByteCodeExpr;
use common::{CrustyError, Field, TableSchema, Tuple};
use core::hash;
use std::collections::HashMap;

/// Hash equi-join implementation. (You can add any other fields that you think are neccessary)
/// A hash equality join operator that builds a hash table from the left child
/// and probes it with tuples from the right child.
pub struct HashEqJoin {
    // Static objects (no need to reset on close)
    managers: &'static Managers,

    // Parameters (no need to reset on close)
    schema: TableSchema,
    left_expr: ByteCodeExpr,
    right_expr: ByteCodeExpr,
    left_child: Box<dyn OpIterator>,
    right_child: Box<dyn OpIterator>,

    /// Hash table mapping a left-side join key to all left tuples with that key.
    hash_map: HashMap<Field, Vec<Tuple>>,

    /// Iterator over the current bucket of left tuples matching the current right tuple.
    current_bucket_iter: Option<std::vec::IntoIter<Tuple>>,

    /// The right-side tuple currently being probed against the hash table.
    current_right_tuple: Option<Tuple>,

    /// Whether the build phase (consuming all left child tuples) is complete.
    hash_map_complete: bool,

    /// Whether the operator has been opened.
    open: bool,
}

impl HashEqJoin {
    /// Creates a new HashEqJoin operator node.
    ///
    /// # Arguments
    /// * `managers` - Reference to the global managers.
    /// * `schema` - The output schema of this join operator.
    /// * `left_expr` - Expression to extract the join key from left-side tuples.
    /// * `right_expr` - Expression to extract the join key from right-side tuples.
    /// * `left_child` - The left child operator (used for the build phase).
    /// * `right_child` - The right child operator (used for the probe phase).
    pub fn new(
        managers: &'static Managers,
        schema: TableSchema,
        left_expr: ByteCodeExpr,
        right_expr: ByteCodeExpr,
        left_child: Box<dyn OpIterator>,
        right_child: Box<dyn OpIterator>,
    ) -> Self {
        HashEqJoin {
            managers,
            left_expr,
            right_expr,
            left_child,
            right_child,
            schema,
            hash_map: HashMap::new(),
            hash_map_complete: false,
            current_bucket_iter: None,
            current_right_tuple: None,
            open: false,
        }
    }
}

impl OpIterator for HashEqJoin {
    fn configure(&mut self, will_rewind: bool) {
        // The left child is fully consumed during the build phase and never rewound
        self.left_child.configure(false);
        self.right_child.configure(will_rewind);
    }

    fn open(&mut self) -> Result<(), CrustyError> {
        self.left_child.open()?;
        self.right_child.open()?;
        self.open = true;
        Ok(())
    }

    /// Advances the iterator, returning the next joined output tuple.
    ///
    /// On the first call, completes the build phase by consuming all left child tuples
    /// into the hash table. Subsequent calls probe the hash table with right-side tuples.
    fn next(&mut self) -> Result<Option<Tuple>, CrustyError> {
        if !self.open {
            panic!("Operator has not been opened");
        }

        loop {
            // Phase 1: Build — consume all left child tuples into the hash table
            while !self.hash_map_complete {
                match self.left_child.next()? {
                    Some(left_tuple) => {
                        let join_key = self.left_expr.eval(&left_tuple);
                        self.hash_map
                            .entry(join_key)
                            .or_insert_with(Vec::new)
                            .push(left_tuple);
                    }
                    None => self.hash_map_complete = true,
                }
            }

            // Phase 2: Probe — if we are mid-bucket, return the next matching joined tuple
            if let Some(ref mut bucket_iter) = self.current_bucket_iter {
                if let Some(left_tuple) = bucket_iter.next() {
                    // Merge the matching left tuple with the current right tuple
                    let joined_tuple = left_tuple.merge(self.current_right_tuple.as_ref().unwrap());
                    return Ok(Some(joined_tuple));
                } else {
                    // Current bucket exhausted; advance to the next right tuple
                    self.current_bucket_iter = None;
                    self.current_right_tuple = None;
                }
            }

            // Phase 3: Advance — fetch the next right tuple and look up its hash bucket
            if self.current_right_tuple.is_none() {
                match self.right_child.next()? {
                    Some(right_tuple) => {
                        let probe_key = self.right_expr.eval(&right_tuple);
                        self.current_right_tuple = Some(right_tuple);

                        if let Some(matching_bucket) = self.hash_map.get(&probe_key) {
                            // Clone the bucket to create an owned iterator over matching left tuples
                            self.current_bucket_iter =
                                Some(matching_bucket.clone().into_iter());
                        } else {
                            // No matching left tuples; discard this right tuple and continue
                            self.current_right_tuple = None;
                        }
                    }
                    // Right child exhausted; join is complete
                    None => return Ok(None),
                }
            }
        }
    }

    fn close(&mut self) -> Result<(), CrustyError> {
        self.left_child.close()?;
        self.right_child.close()?;
        self.open = false;
        Ok(())
    }

    fn rewind(&mut self) -> Result<(), CrustyError> {
        // Only the right child needs to rewind; the left hash table remains intact
        self.right_child.rewind()?;
        Ok(())
    }

    fn get_schema(&self) -> &TableSchema {
        &self.schema
    }
}

#[cfg(test)]
mod test {
    use super::super::TupleIterator;
    use super::*;
    use crate::testutil::execute_iter;
    use crate::testutil::new_test_managers;
    use crate::testutil::TestTuples;
    use common::bytecode_expr::{ByteCodeExpr, ByteCodes};

    fn get_join_predicate() -> (ByteCodeExpr, ByteCodeExpr) {
        // Joining two tables each containing the following tuples:
        // 1 1 3 E
        // 2 1 3 G
        // 3 1 4 A
        // 4 2 4 G
        // 5 2 5 G
        // 6 2 5 G

        // left(col(0) + col(1)) OP right(col(2))
        let mut left = ByteCodeExpr::new();
        left.add_code(ByteCodes::PushField as usize);
        left.add_code(0);
        left.add_code(ByteCodes::PushField as usize);
        left.add_code(1);
        left.add_code(ByteCodes::Add as usize);

        let mut right = ByteCodeExpr::new();
        right.add_code(ByteCodes::PushField as usize);
        right.add_code(2);

        (left, right)
    }

    fn get_iter(left_expr: ByteCodeExpr, right_expr: ByteCodeExpr) -> Box<dyn OpIterator> {
        let setup = TestTuples::new("");
        let managers = new_test_managers();
        let mut iter = Box::new(HashEqJoin::new(
            managers,
            setup.schema.clone(),
            left_expr,
            right_expr,
            Box::new(TupleIterator::new(
                setup.tuples.clone(),
                setup.schema.clone(),
            )),
            Box::new(TupleIterator::new(
                setup.tuples.clone(),
                setup.schema.clone(),
            )),
        ));
        iter.configure(false);
        iter
    }

    fn run_hash_eq_join(left_expr: ByteCodeExpr, right_expr: ByteCodeExpr) -> Vec<Tuple> {
        let mut iter = get_iter(left_expr, right_expr);
        execute_iter(&mut *iter, true).unwrap()
    }

    mod hash_eq_join_test {
        use super::*;

        #[test]
        #[should_panic]
        fn test_empty_predicate_join() {
            let left_expr = ByteCodeExpr::new();
            let right_expr = ByteCodeExpr::new();
            let _ = run_hash_eq_join(left_expr, right_expr);
        }

        #[test]
        fn test_join() {
            // Joining two tables each containing the following tuples:
            // 1 1 3 E
            // 2 1 3 G
            // 3 1 4 A
            // 4 2 4 G
            // 5 2 5 G
            // 6 2 5 G

            // left(col(0) + col(1)) == right(col(2))

            // Output:
            // 2 1 3 G 1 1 3 E
            // 2 1 3 G 2 1 3 G
            // 3 1 4 A 3 1 4 A
            // 3 1 4 A 4 2 4 G
            let (left_expr, right_expr) = get_join_predicate();
            let t = run_hash_eq_join(left_expr, right_expr);
            assert_eq!(t.len(), 4);
            assert_eq!(
                t[0],
                Tuple::new(vec![
                    Field::Int(2),
                    Field::Int(1),
                    Field::Int(3),
                    Field::String("G".to_string()),
                    Field::Int(1),
                    Field::Int(1),
                    Field::Int(3),
                    Field::String("E".to_string()),
                ])
            );
            assert_eq!(
                t[1],
                Tuple::new(vec![
                    Field::Int(2),
                    Field::Int(1),
                    Field::Int(3),
                    Field::String("G".to_string()),
                    Field::Int(2),
                    Field::Int(1),
                    Field::Int(3),
                    Field::String("G".to_string()),
                ])
            );
            assert_eq!(
                t[2],
                Tuple::new(vec![
                    Field::Int(3),
                    Field::Int(1),
                    Field::Int(4),
                    Field::String("A".to_string()),
                    Field::Int(3),
                    Field::Int(1),
                    Field::Int(4),
                    Field::String("A".to_string()),
                ])
            );
            assert_eq!(
                t[3],
                Tuple::new(vec![
                    Field::Int(3),
                    Field::Int(1),
                    Field::Int(4),
                    Field::String("A".to_string()),
                    Field::Int(4),
                    Field::Int(2),
                    Field::Int(4),
                    Field::String("G".to_string()),
                ])
            );
        }
    }

    mod opiterator_test {
        use super::*;
        #[test]
        #[should_panic]
        fn test_next_not_open() {
            let (left_expr, right_expr) = get_join_predicate();
            let mut iter = get_iter(left_expr, right_expr);
            let _ = iter.next();
        }

        #[test]
        #[should_panic]
        fn test_rewind_not_open() {
            let (left_expr, right_expr) = get_join_predicate();
            let mut iter = get_iter(left_expr, right_expr);
            let _ = iter.rewind();
        }

        #[test]
        fn test_open() {
            let (left_expr, right_expr) = get_join_predicate();
            let mut iter = get_iter(left_expr, right_expr);
            iter.open().unwrap();
        }

        #[test]
        fn test_close() {
            let (left_expr, right_expr) = get_join_predicate();
            let mut iter = get_iter(left_expr, right_expr);
            iter.open().unwrap();
            iter.close().unwrap();
        }

        #[test]
        fn test_rewind() {
            let (left_expr, right_expr) = get_join_predicate();
            let mut iter = get_iter(left_expr, right_expr);
            iter.configure(true);
            let t_before = execute_iter(&mut *iter, false).unwrap();
            iter.rewind().unwrap();
            let t_after = execute_iter(&mut *iter, false).unwrap();
            assert_eq!(t_before, t_after);
        }
    }
}

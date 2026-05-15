use super::OpIterator;

use common::bytecode_expr::ByteCodeExpr;
use common::datatypes::compare_fields;
use common::{BooleanOp, CrustyError, TableSchema, Tuple};

/// Nested loop join implementation. (You can add any other fields that you think are neccessary)
/// A nested loop join operator that evaluates a boolean join condition
/// by iterating the right child once for every tuple in the left child.
pub struct NestedLoopJoin {
    // Parameters (no need to reset on close)
    schema: TableSchema,
    /// The boolean comparison operation used in the join predicate.
    op: BooleanOp,
    /// Expression to extract the join key from left-side tuples.
    left_expr: ByteCodeExpr,
    /// Expression to extract the join key from right-side tuples.
    right_expr: ByteCodeExpr,
    left_child: Box<dyn OpIterator>,
    right_child: Box<dyn OpIterator>,

    /// The current left-side tuple being probed against the entire right child.
    current_left_tuple: Option<Tuple>,
    /// Whether the operator has been opened.
    open: bool,
}

impl NestedLoopJoin {
    /// Creates a new NestedLoopJoin operator node.
    ///
    /// # Arguments
    /// * `op` - The boolean comparison operation for the join predicate.
    /// * `left_expr` - Expression to extract the join key from left-side tuples.
    /// * `right_expr` - Expression to extract the join key from right-side tuples.
    /// * `left_child` - The left (outer) child operator.
    /// * `right_child` - The right (inner) child operator, rewound for each left tuple.
    /// * `schema` - The output schema of this join operator.
    pub fn new(
        op: BooleanOp,
        left_expr: ByteCodeExpr,
        right_expr: ByteCodeExpr,
        left_child: Box<dyn OpIterator>,
        right_child: Box<dyn OpIterator>,
        schema: TableSchema,
    ) -> Self {
        NestedLoopJoin {
            op,
            left_expr,
            right_expr,
            left_child,
            right_child,
            schema,
            current_left_tuple: None,
            open: false,
        }
    }
}

impl OpIterator for NestedLoopJoin {
    fn configure(&mut self, will_rewind: bool) {
        self.left_child.configure(will_rewind);
        // The right child is always rewound after each left tuple, so it must support rewind
        self.right_child.configure(true);
    }

    fn open(&mut self) -> Result<(), CrustyError> {
        self.left_child.open()?;
        self.right_child.open()?;
        self.open = true;
        Ok(())
    }

    /// Advances the iterator, returning the next joined output tuple.
    ///
    /// For each left tuple, scans the entire right child looking for tuples
    /// that satisfy the join predicate. When the right child is exhausted,
    /// it is rewound and the next left tuple is fetched.
    fn next(&mut self) -> Result<Option<Tuple>, CrustyError> {
        if !self.open {
            panic!("Operator has not been opened");
        }

        loop {
            // Fetch the next left (outer) tuple if we don't have one
            if self.current_left_tuple.is_none() {
                match self.left_child.next()? {
                    Some(left_tuple) => self.current_left_tuple = Some(left_tuple),
                    // Left child exhausted; join is complete
                    None => return Ok(None),
                }
            }

            // Scan the right child for tuples satisfying the join predicate
            while let Some(right_tuple) = self.right_child.next()? {
                let left_key = self.left_expr.eval(self.current_left_tuple.as_ref().unwrap());
                let right_key = self.right_expr.eval(&right_tuple);

                if compare_fields(self.op, &left_key, &right_key) {
                    // Predicate satisfied; merge and return the joined tuple
                    let joined_tuple = self.current_left_tuple.as_ref().unwrap().merge(&right_tuple);
                    return Ok(Some(joined_tuple));
                }
            }

            // Right child exhausted for this left tuple; rewind and advance the left child
            self.right_child.rewind()?;
            self.current_left_tuple = None;
        }
    }

    fn close(&mut self) -> Result<(), CrustyError> {
        self.left_child.close()?;
        self.right_child.close()?;
        self.open = false;
        Ok(())
    }

    fn rewind(&mut self) -> Result<(), CrustyError> {
        self.left_child.rewind()?;
        self.right_child.rewind()?;
        Ok(())
    }

    /// Returns the output schema of this join operator.
    fn get_schema(&self) -> &TableSchema {
        &self.schema
    }
}
#[cfg(test)]
mod test {
    use super::super::TupleIterator;
    use super::*;
    use crate::testutil::execute_iter;
    use crate::testutil::TestTuples;
    use common::bytecode_expr::{ByteCodeExpr, ByteCodes};
    use common::Field;

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

    fn get_iter(
        op: BooleanOp,
        left_expr: ByteCodeExpr,
        right_expr: ByteCodeExpr,
    ) -> Box<dyn OpIterator> {
        let setup = TestTuples::new("");
        let mut iter = Box::new(NestedLoopJoin::new(
            op,
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
            setup.schema.clone(),
        ));
        iter.configure(false);
        iter
    }

    fn run_nested_loop_join(
        op: BooleanOp,
        left_expr: ByteCodeExpr,
        right_expr: ByteCodeExpr,
    ) -> Vec<Tuple> {
        let mut iter = get_iter(op, left_expr, right_expr);
        execute_iter(&mut *iter, true).unwrap()
    }

    mod nested_loop_join_test {
        use super::*;

        #[test]
        #[should_panic]
        fn test_empty_predicate_join() {
            let left_expr = ByteCodeExpr::new();
            let right_expr = ByteCodeExpr::new();
            let _ = run_nested_loop_join(BooleanOp::Eq, left_expr, right_expr);
        }

        #[test]
        fn test_eq_join() {
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
            let t = run_nested_loop_join(BooleanOp::Eq, left_expr, right_expr);
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
            let mut iter = get_iter(BooleanOp::Eq, left_expr, right_expr);
            let _ = iter.next();
        }

        #[test]
        #[should_panic]
        fn test_rewind_not_open() {
            let (left_expr, right_expr) = get_join_predicate();
            let mut iter = get_iter(BooleanOp::Eq, left_expr, right_expr);
            let _ = iter.rewind();
        }

        #[test]
        fn test_open() {
            let (left_expr, right_expr) = get_join_predicate();
            let mut iter = get_iter(BooleanOp::Eq, left_expr, right_expr);
            iter.open().unwrap();
        }

        #[test]
        fn test_close() {
            let (left_expr, right_expr) = get_join_predicate();
            let mut iter = get_iter(BooleanOp::Eq, left_expr, right_expr);
            iter.open().unwrap();
            iter.close().unwrap();
        }

        #[test]
        fn test_rewind() {
            let (left_expr, right_expr) = get_join_predicate();
            let mut iter = get_iter(BooleanOp::Eq, left_expr, right_expr);
            iter.configure(true);
            let t_before = execute_iter(&mut *iter, false).unwrap();
            iter.rewind().unwrap();
            let t_after = execute_iter(&mut *iter, false).unwrap();
            assert_eq!(t_before, t_after);
        }
    }
}

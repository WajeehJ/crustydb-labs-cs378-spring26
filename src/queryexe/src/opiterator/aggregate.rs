use super::OpIterator;
use crate::Managers;
use common::bytecode_expr::ByteCodeExpr;
use common::datatypes::f_decimal;
use common::{AggOp, CrustyError, Field, TableSchema, Tuple};
use std::cmp::{max, min};
use std::collections::HashMap;
use std::collections::hash_map::IntoIter;

/// Aggregate operator. (You can add any other fields that you think are neccessary)
/// 
/// 
pub struct Aggregate {
    // Static objects (No need to reset on close)
    managers: &'static Managers,

    // Parameters (No need to reset on close)
    /// Output schema of the form [groupby_field attributes ..., agg_field attributes ...]).
    schema: TableSchema,
    /// Group by fields
    groupby_expr: Vec<ByteCodeExpr>,
    /// Aggregated fields.
    agg_expr: Vec<ByteCodeExpr>,
    /// Aggregation operations.
    ops: Vec<AggOp>,
    /// Child operator to get the data from.
    child: Box<dyn OpIterator>,
    /// If true, then the operator will be rewinded in the future.
    will_rewind: bool,
    hash_map: HashMap<Vec<Field>, Vec<AggState>>,
    current_list_iter: Option<IntoIter<Vec<Field>, Vec<AggState>>>,
    open: bool,
    results: Option<Vec<Tuple>>, 
    cursor: usize,
    
    // States (Need to reset on close)
    // todo!("Your code here")
}

/// Represents the state of an aggregation operation during processing.
/// 
/// - `Value`: Used for Min, Max, Sum, and Count operations, storing the current accumulated field.
/// - `AvgState`: Used for Average operations, storing the running sum and the count of values seen.
enum AggState {
    Value(Field),
    AvgState(Field, i64),
}

/// Scale factor used when converting a floating-point average to a fixed-point Decimal field.
/// Represents 4 decimal places of precision (i.e., multiply by 10^4 = 10000).
const AVG_DECIMAL_SCALE: u32 = 4;
const AVG_DECIMAL_MULTIPLIER: f64 = 10_000.0;

impl Aggregate {
    /// Creates a new Aggregate operator.
    ///
    /// # Arguments
    /// * `managers` - Reference to the global managers.
    /// * `groupby_expr` - Expressions used to compute the group-by key for each tuple.
    /// * `agg_expr` - Expressions used to extract the value to aggregate from each tuple.
    /// * `ops` - Aggregation operations (e.g., Sum, Avg) corresponding to each `agg_expr`.
    /// * `schema` - The output schema of this operator.
    /// * `child` - The child operator that produces input tuples.
    pub fn new(
        managers: &'static Managers,
        groupby_expr: Vec<ByteCodeExpr>,
        agg_expr: Vec<ByteCodeExpr>,
        ops: Vec<AggOp>,
        schema: TableSchema,
        child: Box<dyn OpIterator>,
    ) -> Self {
        assert!(
            ops.len() == agg_expr.len(),
            "ops and agg_expr must have the same length"
        );
        Aggregate {
            managers,
            schema,
            groupby_expr,
            agg_expr,
            ops,
            child,
            will_rewind: true,
            hash_map: HashMap::new(),
            open: false,
            current_list_iter: None,
            results: None,
            cursor: 0,
        }
    }

    /// Merges a single field value into an accumulator state for a given aggregation operation.
    ///
    /// Null field values are ignored (treated as no-ops) per standard SQL aggregation semantics.
    ///
    /// # Arguments
    /// * `op` - The aggregation operation to apply.
    /// * `field_val` - The new field value to incorporate.
    /// * `acc` - The mutable accumulator state to update.
    ///
    /// # Errors
    /// Returns a `CrustyError` if the field arithmetic operation fails.
    fn merge_fields(
        op: AggOp,
        field_val: &Field,
        acc: &mut AggState,
    ) -> Result<(), CrustyError> {
        // SQL semantics: null values are ignored in aggregations
        if let Field::Null = field_val {
            return Ok(());
        }

        match (op, acc) {
            (AggOp::Min, AggState::Value(acc_val)) => {
                if let Field::Null = acc_val {
                    *acc_val = field_val.clone();
                } else if field_val < acc_val {
                    *acc_val = field_val.clone();
                }
            }

            (AggOp::Max, AggState::Value(acc_val)) => {
                if let Field::Null = acc_val {
                    *acc_val = field_val.clone();
                } else if field_val > acc_val {
                    *acc_val = field_val.clone();
                }
            }

            (AggOp::Sum, AggState::Value(acc_val)) => {
                if let Field::Null = acc_val {
                    *acc_val = field_val.clone();
                } else {
                    *acc_val = (acc_val.clone() + field_val.clone())?;
                }
            }

            (AggOp::Avg, AggState::AvgState(running_sum, count)) => {
                if let Field::Null = running_sum {
                    *running_sum = field_val.clone();
                } else {
                    *running_sum = (running_sum.clone() + field_val.clone())?;
                }
                *count += 1;
            }

            (AggOp::Count, AggState::Value(count_field)) => {
                if let Field::Null = count_field {
                    *count_field = Field::Int(1);
                } else {
                    *count_field = (count_field.clone() + Field::Int(1))?;
                }
            }

            _ => panic!("Mismatched AggOp and AggState variant"),
        }
        Ok(())
    }

    /// Evaluates the group-by key and aggregation expressions for a tuple,
    /// then merges the result into the appropriate group entry in the hash map.
    ///
    /// # Arguments
    /// * `tuple` - The input tuple to incorporate into a group.
    pub fn merge_tuple_into_group(&mut self, tuple: &Tuple) {
        // Compute the group-by key by evaluating each group-by expression
        let group_key: Vec<Field> = self
            .groupby_expr
            .iter()
            .map(|expr| expr.eval(tuple))
            .collect();

        // Initialize accumulator states for a new group, or retrieve the existing ones
        let agg_states = self.hash_map.entry(group_key).or_insert_with(|| {
            self.ops
                .iter()
                .map(|op| match op {
                    // Avg tracks sum and count separately
                    AggOp::Avg => AggState::AvgState(Field::Int(0), 0),
                    // All other ops start with a null accumulator
                    _ => AggState::Value(Field::Null),
                })
                .collect()
        });

        // Merge each aggregation expression's value into its corresponding state
        for (index, op) in self.ops.iter().enumerate() {
            let field_val = self.agg_expr[index].eval(tuple);
            let state = &mut agg_states[index];
            Self::merge_fields(*op, &field_val, state).expect("Merge failed");
        }
    }
}

impl OpIterator for Aggregate {
    fn configure(&mut self, will_rewind: bool) {
        self.will_rewind = will_rewind;
        // The child will never be rewound because the aggregate buffers all child tuples
        self.child.configure(false);
    }

    fn open(&mut self) -> Result<(), CrustyError> {
        self.child.open()?;
        self.open = true;
        Ok(())
    }

    /// Advances the iterator, returning the next aggregated output tuple.
    ///
    /// On the first call, consumes all child tuples and materializes the aggregation results.
    /// Subsequent calls iterate over the buffered results.
    fn next(&mut self) -> Result<Option<Tuple>, CrustyError> {
        // Materialize results on the first call to next()
        if self.results.is_none() {
            // Consume all tuples from the child and accumulate into groups
            while let Some(child_tuple) = self.child.next()? {
                self.merge_tuple_into_group(&child_tuple);
            }

            // Convert each group's accumulated state into a final output tuple
            let mut final_tuples = Vec::new();
            for (group_key_fields, agg_states) in std::mem::take(&mut self.hash_map) {
                // Start with the group-by key fields, then append aggregated values
                let mut output_row = group_key_fields;

                for state in agg_states {
                    match state {
                        AggState::Value(value) => output_row.push(value),

                        AggState::AvgState(running_sum, count) => {
                            // Convert the running sum to f64 for division
                            let sum_as_f64 = match running_sum {
                                Field::Int(integer_val) => integer_val as f64,
                                Field::Decimal(raw_val, scale) => {
                                    // Normalize fixed-point decimal to f64
                                    raw_val as f64 / 10f64.powi(scale as i32)
                                }
                                _ => 0.0,
                            };

                            let average = sum_as_f64 / (count as f64);

                            // Store as a fixed-point Decimal with AVG_DECIMAL_SCALE precision
                            output_row.push(Field::Decimal(
                                (average * AVG_DECIMAL_MULTIPLIER) as i64,
                                AVG_DECIMAL_SCALE,
                            ));
                        }
                    }
                }

                final_tuples.push(Tuple::new(output_row));
            }

            self.results = Some(final_tuples);
            self.cursor = 0;
        }

        // Return the next buffered result tuple, or None if exhausted
        let results_vec = self.results.as_ref().unwrap();
        if self.cursor < results_vec.len() {
            let next_tuple = results_vec[self.cursor].clone();
            self.cursor += 1;
            Ok(Some(next_tuple))
        } else {
            Ok(None)
        }
    }

    fn close(&mut self) -> Result<(), CrustyError> {
        self.child.close()?;
        self.hash_map.clear();
        self.current_list_iter = None;
        self.open = false;
        Ok(())
    }

    fn rewind(&mut self) -> Result<(), CrustyError> {
        if !self.open {
            panic!("Tried rewinding while closed");
        }
        // Reset cursor to replay buffered results from the beginning
        self.cursor = 0;
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
    use crate::testutil::{execute_iter, new_test_managers, TestTuples};
    use common::{
        bytecode_expr::colidx_expr,
        datatypes::{f_int, f_str},
    };

    fn get_iter(
        groupby_expr: Vec<ByteCodeExpr>,
        agg_expr: Vec<ByteCodeExpr>,
        ops: Vec<AggOp>,
    ) -> Box<dyn OpIterator> {
        let setup = TestTuples::new("");
        let managers = new_test_managers();
        let dummy_schema = TableSchema::new(vec![]);
        let mut iter = Box::new(Aggregate::new(
            managers,
            groupby_expr,
            agg_expr,
            ops,
            dummy_schema,
            Box::new(TupleIterator::new(
                setup.tuples.clone(),
                setup.schema.clone(),
            )),
        ));
        iter.configure(false);
        iter
    }

    fn run_aggregate(
        groupby_expr: Vec<ByteCodeExpr>,
        agg_expr: Vec<ByteCodeExpr>,
        ops: Vec<AggOp>,
    ) -> Vec<Tuple> {
        let mut iter = get_iter(groupby_expr, agg_expr, ops);
        execute_iter(&mut *iter, true).unwrap()
    }

    mod aggregation_test {
        use super::*;

        #[test]
        fn test_empty_group() {
            let group_by = vec![];
            let agg = vec![colidx_expr(0), colidx_expr(1), colidx_expr(2)];
            let ops = vec![AggOp::Count, AggOp::Max, AggOp::Avg];
            let t = run_aggregate(group_by, agg, ops);
            assert_eq!(t.len(), 1);
            assert_eq!(t[0], Tuple::new(vec![f_int(6), f_int(2), f_decimal(4.0)]));
        }

        #[test]
        fn test_empty_aggregation() {
            let group_by = vec![colidx_expr(2)];
            let agg = vec![];
            let ops = vec![];
            let t = run_aggregate(group_by, agg, ops);
            assert_eq!(t.len(), 3);
            assert_eq!(t[0], Tuple::new(vec![f_int(3)]));
            assert_eq!(t[1], Tuple::new(vec![f_int(4)]));
            assert_eq!(t[2], Tuple::new(vec![f_int(5)]));
        }

        #[test]
        fn test_count() {
            // Input:
            // 1 1 3 E
            // 2 1 3 G
            // 3 1 4 A
            // 4 2 4 G
            // 5 2 5 G
            // 6 2 5 G
            let group_by = vec![colidx_expr(1), colidx_expr(2)];
            let agg = vec![colidx_expr(0)];
            let ops = vec![AggOp::Count];
            let t = run_aggregate(group_by, agg, ops);
            // Output:
            // 1 3 2
            // 1 4 1
            // 2 4 1
            // 2 5 2
            assert_eq!(t.len(), 4);
            assert_eq!(t[0], Tuple::new(vec![f_int(1), f_int(3), f_int(2)]));
            assert_eq!(t[1], Tuple::new(vec![f_int(1), f_int(4), f_int(1)]));
            assert_eq!(t[2], Tuple::new(vec![f_int(2), f_int(4), f_int(1)]));
            assert_eq!(t[3], Tuple::new(vec![f_int(2), f_int(5), f_int(2)]));
        }

        #[test]
        fn test_sum() {
            // Input:
            // 1 1 3 E
            // 2 1 3 G
            // 3 1 4 A
            // 4 2 4 G
            // 5 2 5 G
            // 6 2 5 G

            let group_by = vec![colidx_expr(1), colidx_expr(2)];
            let agg = vec![colidx_expr(0)];
            let ops = vec![AggOp::Sum];
            let tuples = run_aggregate(group_by, agg, ops);
            // Output:
            // 1 3 3
            // 1 4 3
            // 2 4 4
            // 2 5 11
            assert_eq!(tuples.len(), 4);
            assert_eq!(tuples[0], Tuple::new(vec![f_int(1), f_int(3), f_int(3)]));
            assert_eq!(tuples[1], Tuple::new(vec![f_int(1), f_int(4), f_int(3)]));
            assert_eq!(tuples[2], Tuple::new(vec![f_int(2), f_int(4), f_int(4)]));
            assert_eq!(tuples[3], Tuple::new(vec![f_int(2), f_int(5), f_int(11)]));
        }

        #[test]
        fn test_max() {
            // Input:
            // 1 1 3 E
            // 2 1 3 G
            // 3 1 4 A
            // 4 2 4 G
            // 5 2 5 G
            // 6 2 5 G

            let group_by = vec![colidx_expr(1), colidx_expr(2)];
            let agg = vec![colidx_expr(3)];
            let ops = vec![AggOp::Max];
            let t = run_aggregate(group_by, agg, ops);
            // Output:
            // 1 3 G
            // 1 4 A
            // 2 4 G
            // 2 5 G
            assert_eq!(t.len(), 4);
            assert_eq!(t[0], Tuple::new(vec![f_int(1), f_int(3), f_str("G")]));
            assert_eq!(t[1], Tuple::new(vec![f_int(1), f_int(4), f_str("A")]));
            assert_eq!(t[2], Tuple::new(vec![f_int(2), f_int(4), f_str("G")]));
            assert_eq!(t[3], Tuple::new(vec![f_int(2), f_int(5), f_str("G")]));
        }

        #[test]
        fn test_min() {
            // Input:
            // 1 1 3 E
            // 2 1 3 G
            // 3 1 4 A
            // 4 2 4 G
            // 5 2 5 G
            // 6 2 5 G

            let group_by = vec![colidx_expr(1), colidx_expr(2)];
            let agg = vec![colidx_expr(3)];
            let ops = vec![AggOp::Min];
            let t = run_aggregate(group_by, agg, ops);
            // Output:
            // 1 3 E
            // 1 4 A
            // 2 4 G
            // 2 5 G
            assert!(t.len() == 4);
            assert_eq!(t[0], Tuple::new(vec![f_int(1), f_int(3), f_str("E")]));
            assert_eq!(t[1], Tuple::new(vec![f_int(1), f_int(4), f_str("A")]));
            assert_eq!(t[2], Tuple::new(vec![f_int(2), f_int(4), f_str("G")]));
            assert_eq!(t[3], Tuple::new(vec![f_int(2), f_int(5), f_str("G")]));
        }

        #[test]
        fn test_avg() {
            // Input:
            // 1 1 3 E
            // 2 1 3 G
            // 3 1 4 A
            // 4 2 4 G
            // 5 2 5 G
            // 6 2 5 G
            let group_by = vec![colidx_expr(1), colidx_expr(2)];
            let agg = vec![colidx_expr(0)];
            let ops = vec![AggOp::Avg];
            let t = run_aggregate(group_by, agg, ops);
            // Output:
            // 1 3 1.5
            // 1 4 3.0
            // 2 4 4.0
            // 2 5 5.5
            assert_eq!(t.len(), 4);
            assert_eq!(t[0], Tuple::new(vec![f_int(1), f_int(3), f_decimal(1.5)]));
            assert_eq!(t[1], Tuple::new(vec![f_int(1), f_int(4), f_decimal(3.0)]));
            assert_eq!(t[2], Tuple::new(vec![f_int(2), f_int(4), f_decimal(4.0)]));
            assert_eq!(t[3], Tuple::new(vec![f_int(2), f_int(5), f_decimal(5.5)]));
        }

        #[test]
        fn test_multi_column_aggregation() {
            // Input:
            // 1 1 3 E
            // 2 1 3 G
            // 3 1 4 A
            // 4 2 4 G
            // 5 2 5 G
            // 6 2 5 G
            let group_by = vec![colidx_expr(3)];
            let agg = vec![colidx_expr(0), colidx_expr(1), colidx_expr(2)];
            let ops = vec![AggOp::Count, AggOp::Max, AggOp::Avg];
            let t = run_aggregate(group_by, agg, ops);
            // Output:
            // A 1 1 4.0
            // E 1 1 3.0
            // G 4 2 4.25
            assert_eq!(t.len(), 3);
            assert_eq!(
                t[0],
                Tuple::new(vec![f_str("A"), f_int(1), f_int(1), f_decimal(4.0)])
            );
            assert_eq!(
                t[1],
                Tuple::new(vec![f_str("E"), f_int(1), f_int(1), f_decimal(3.0)])
            );
            assert_eq!(
                t[2],
                Tuple::new(vec![f_str("G"), f_int(4), f_int(2), f_decimal(4.25)])
            );
        }

        #[test]
        #[should_panic]
        fn test_merge_tuples_not_int() {
            let group_by = vec![];
            let agg = vec![colidx_expr(3)];
            let ops = vec![AggOp::Avg];
            let _ = run_aggregate(group_by, agg, ops);
        }
    }

    mod opiterator_test {
        use super::*;

        #[test]
        #[should_panic]
        fn test_next_not_open() {
            let mut iter = get_iter(vec![], vec![], vec![]);
            let _ = iter.next();
        }

        #[test]
        #[should_panic]
        fn test_rewind_not_open() {
            let mut iter = get_iter(vec![], vec![], vec![]);
            let _ = iter.rewind();
        }

        #[test]
        fn test_open() {
            let mut iter = get_iter(vec![], vec![], vec![]);
            iter.open().unwrap();
        }

        #[test]
        fn test_close() {
            let mut iter = get_iter(vec![], vec![], vec![]);
            iter.open().unwrap();
            iter.close().unwrap();
        }

        #[test]
        fn test_rewind() {
            let mut iter = get_iter(vec![colidx_expr(2)], vec![colidx_expr(0)], vec![AggOp::Max]);
            iter.configure(true); // if we will rewind in the future, then we set will_rewind to true
            let t_before = execute_iter(&mut *iter, true).unwrap();
            iter.rewind().unwrap();
            let t_after = execute_iter(&mut *iter, true).unwrap();
            assert_eq!(t_before, t_after);
        }
    }
}

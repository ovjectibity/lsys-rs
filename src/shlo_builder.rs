use xla::{
    XlaBuilder,
    XlaOp,
    XlaComputation
};

fn build_ops() {
    let ops_graph = XlaBuilder::new("init");
    let op1 = XlaOp::abs(&self);
    // XlaOp::reduce(&self, init_value, comp, dims, keep_dims)
    // XlaOp::mul_(&self, op)
    // ops_graph.build(op1);
}
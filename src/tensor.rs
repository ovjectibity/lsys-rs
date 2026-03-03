// use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

pub struct Tensor {
    shape: Vec<i64>,
    dim: i64, 
    size: i64,
    data: Rc<RefCell<Vec<f64>>>
}

impl Tensor {
    
}
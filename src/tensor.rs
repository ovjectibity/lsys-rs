// use std::collections::HashMap;
// use std::rc::Rc;
// use std::cell::RefCell;

trait ComputationNode {
    fn forward(&mut self);
    fn backward(&mut self);
    fn get_activations(&self) -> Vec<f64>;

    fn change_params(&mut self) {

    }

    //Returns [l_size, num_size] Jacobian
    fn get_gradient(&self) -> Vec<Vec<f64>>;

    //Returns [num_size, num_parent_size] Jacobian
    fn get_gradient_wrt_parent(&self,index: i64) -> Vec<Vec<f64>>;

    fn get_l_size(&self) -> i64; 
}

trait Optimiser {
    fn step();
}

trait TensorFunction {
    //Returns num_size vec
    fn apply(&self,args: &Vec<Tensor>, 
        params: &Vec<f64>) -> Vec<f64>;
    //Returns [num_size, num_parent_size] Jacobian
    fn gradient_wrt_parent(&self,args: &Vec<Tensor>,
        index: i64, 
        params: &Vec<f64>) -> Vec<Vec<f64>>;
    //Returns [num_size, num_params] Jacobian
    fn gradient_wrt_params(&self,args: &Vec<Tensor>, 
        params: &Vec<f64>) -> Vec<Vec<f64>>;
}

//Represents a tensor & a computational graph: 
struct Tensor {
    id: i64,
    dim: i64,
    //This is the sum of all elements of the shape
    size: i64,
    shape: Vec<i32>,
    //The parent nodes computing the tensor
    parents: Vec<Tensor>,
    //The children nodes using the tensor
    children: Vec<Tensor>,
    composer: Box<dyn TensorFunction>,
    params: Vec<f64>,
    //Stored in the reverse order of the indices represented by shape
    current_activations: Vec<f64>,
    //Gradients wrt to self activations & params
    //The gradients are always for a specific tensor known as 
    // the superchild & whose size is l_size
    //[l_size,num_size]
    gradients_self: Vec<Vec<f64>>,
    //[l_size,num_params]
    gradients_params: Vec<Vec<f64>>,
    dirty: bool
}

impl Tensor {
    fn add_matrices(m1: &Vec<Vec<f64>>,m2: &Vec<Vec<f64>>) -> 
    Vec<Vec<f64>> {
        vec![vec![0.0]]
    }

    fn multiply_matrices(m1: &Vec<Vec<f64>>,m2: &Vec<Vec<f64>>) -> 
    Vec<Vec<f64>> {
        vec![vec![0.0]]
    }

    fn transpose_matrix(m1: &Vec<Vec<f64>>) ->
    Vec<Vec<f64>> {
        vec![vec![0.0]]
    }

    fn flatten_last_for_matrix(t1: &Vec<Vec<Vec<f64>>>) -> Vec<Vec<f64>> {
        vec![vec![0.0]]
    }
}

impl ComputationNode for Tensor {
    fn forward(&mut self) {
        //TODO: Override computation if already computed:
        if self.dirty {
            return;
        }
        //TODO: Compute this for children rather than the parents??
        for parent in 
        &mut self.parents {
            parent.forward();
        }
        //TODO: Ensure that the shape of vec 
        //returned by the composer is as expected
        self.current_activations = 
            self.composer.apply(&self.parents,&self.params);
    }

    //TODO: Not sure if this is efficient
    fn backward(&mut self) {
        //This function computes gradients with respect to its self activations & params
        //i.e. populates gradients_self & gradients_params: 
        //Computing the gradient_self by using the child's activation gradients 
        //& the gradients of child's activation wrt to parent activations
        let mut acc = vec![vec![
            0.0;self.size as usize];self.get_l_size() as usize];
        for child in &self.children {
            //Returns [l_size, num_child_size] array
            let child_grad_m = 
                child.get_gradient();
            //Returns [num_child_size, num_size] Jacobian
            let child_self_grad = 
                child.get_gradient_wrt_parent(self.id);
            let grad_part = Tensor::multiply_matrices(
                &child_grad_m, &child_self_grad);
            acc = Tensor::add_matrices(&acc, &grad_part);
        }
        
        //[l_size,num_size]
        self.gradients_self = acc;

        //Compute gradients wrt to params now: 
        self.gradients_params = Tensor::multiply_matrices(
            &self.gradients_self,
            &self.composer.gradient_wrt_params(
                &self.parents, &self.params));
    }

    fn get_activations(&self) -> Vec<f64> {
        //TODO: Does this need a clone? 
        self.current_activations.clone()
    }

    fn get_gradient(&self) -> Vec<Vec<f64>> {
        vec![vec![0.0]]
    }

    fn get_gradient_wrt_parent(&self,index: i64) -> Vec<Vec<f64>> {
        self.composer.gradient_wrt_parent(
            &self.parents, index, &self.params)
    }
}
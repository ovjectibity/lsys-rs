// use std::collections::HashMap;
// use std::rc::Rc;
// use std::cell::RefCell;

//NN: 
//Weight matrix multiply & then apply non-linearity
struct FFN {
    //t1 * w1
}

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
    fn apply(&self,
        args: &Vec<Tensor>, 
        params: &Vec<f64>) -> Vec<f64>;
    //Returns [num_size, num_parent_size] Jacobian
    //Uses the parents activations to computie this: 
    fn gradient_wrt_parent(&self,
        args: &Vec<Tensor>,
        index: i64, 
        params: &Vec<f64>) -> Vec<Vec<f64>>;
    //Returns [num_size, num_params] Jacobian
    fn gradient_wrt_params(&self,args: &Vec<Tensor>, 
        params: &Vec<f64>) -> Vec<Vec<f64>>;
}

struct ReluLayer {

}

impl TensorFunction for ReluLayer {
    fn apply(&self,
        args: &Vec<Tensor>, 
        params: &Vec<f64>) -> Vec<f64> {
        assert!(args.len() == 1);
        assert!(args.get(0).expect("").dim == 1);
        assert!(params.len() == 0);
        let act = &args.get(0).expect("").current_activations;
        act.iter().map(|e| {
            if *e > 0.0 {
                *e
            } else {
                0.0
            }
        }).collect()
    }

    fn gradient_wrt_parent(&self,
        args: &Vec<Tensor>,
        index: i64, 
        params: &Vec<f64>) -> Vec<Vec<f64>> {
        assert!(index == 0);
        assert!(args.len() == 1);
        assert!(args.get(0).expect("").dim == 1);
        assert!(params.len() == 0);
        let parent_size = args.get(0).expect("").size;
        let act = &args.get(0).expect("").current_activations;
        let mut acc = Vec::new();
        for e in act.iter().enumerate() {
            let mut grad_part = vec![0.0; parent_size as usize - 1];
            let grad = if *e.1 > 0.0 {
                1.0
            } else {
                0.0
            };
            grad_part.splice(e.0..e.0, [grad]);
            acc.push(grad_part);
        }
        acc
    }

    fn gradient_wrt_params(&self,
        args: &Vec<Tensor>, 
        params: &Vec<f64>) -> Vec<Vec<f64>> {
        vec![]
    }
}

struct LinearLayer {

}

impl TensorFunction for LinearLayer {
    fn apply(&self,
        args: &Vec<Tensor>, 
        params: &Vec<f64>) -> Vec<f64> {
        //Assert that the args is just 1 dim tensor
        assert!(args.len() == 1);
        assert!(args.get(0).expect("").dim == 1);
        assert!(params.len() as i64 % args.get(0).expect("").size == 0);
        let arg_size = args.get(0).expect("").size as i64;
        let args_activations = 
            Tensor::unflatten(&args.get(0).expect("").
                            current_activations,
                            1);
        
        //Params need to be: [num_size,num_arg_size]
        let params_m = 
            Tensor::unflatten(params, arg_size);
        Tensor::flatten(&Tensor::multiply_matrices(
            &params_m, &args_activations))
    }

    fn gradient_wrt_parent(&self,
        args: &Vec<Tensor>,
        index: i64, 
        params: &Vec<f64>) -> Vec<Vec<f64>> {
        assert!(index == 0);
        assert!(args.len() == 1);
        assert!(args.get(0).expect("").dim == 1);
        assert!(params.len() as i64 % args.get(0).expect("").size == 0);
        let arg_size = args.get(0).expect("").size;
        //Return the weight matrix directly: 
        Tensor::unflatten(&params, arg_size)
    }

    //Returns [num_size, num_params] Jacobian
    fn gradient_wrt_params(&self,args: &Vec<Tensor>, 
        params: &Vec<f64>) -> Vec<Vec<f64>> {
        assert!(args.len() == 1);
        assert!(args.get(0).expect("").dim == 1);
        assert!(params.len() as i64 % args.get(0).expect("").size == 0);
        let param_len = params.len() as i64;
        let parent_activations = 
            &args.get(0).expect("").current_activations.clone();
        let num_size = param_len / parent_activations.len() as i64;
        let mut acc = Vec::new();
        for i in 0..num_size {
            let mut grad_params = vec![0.0; param_len as usize - 
                parent_activations.len() as usize];
            grad_params.splice(i as usize..i as usize, 
                parent_activations.iter().cloned());
            acc.push(grad_params);
        }
        acc
    }
}

//Represents a tensor & a computational graph: 
//TODO: Implement shallow copies & management of data flattened 
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
    gradients_params: Vec<Vec<f64>>
}

impl Tensor {
    fn add_matrices(
        m1: &Vec<Vec<f64>>,
        m2: &Vec<Vec<f64>>) -> 
    Vec<Vec<f64>> {
        //TODO: Impl this: 
        // assert!()
        vec![vec![0.0]]
    }

    fn multiply_matrices(m1: &Vec<Vec<f64>>,
        m2: &Vec<Vec<f64>>) -> 
    Vec<Vec<f64>> {
        //TODO: Impl this: 
        vec![vec![0.0]]
    }

    fn transpose_matrix(m1: &Vec<Vec<f64>>) ->
    Vec<Vec<f64>> {
        //TODO: Impl this: 
        vec![vec![0.0]]
    }

    fn flatten_last_for_matrix(
        t1: &Vec<Vec<Vec<f64>>>) -> 
    Vec<Vec<f64>> {
        //TODO: Impl this: 
        vec![vec![0.0]]
    }

    fn flatten(
        t1: &Vec<Vec<f64>>) -> 
    Vec<f64> {
        //TODO: Impl this: 
        vec![0.0]
    }

    //Returns [t1.len() / end_dim, end_dim]
    fn unflatten(
        t1: &Vec<f64>,
        end_dim: i64) -> 
    Vec<Vec<f64>> {
        //TODO: Impl this: 
        vec![vec![0.0]]
    }
}

impl ComputationNode for Tensor {
    fn forward(&mut self) {
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
        let l_size = self.children.get(0).expect("").get_l_size();
        let mut acc = vec![vec![
            0.0;self.size as usize];l_size as usize];
        for child in &mut self.children {
            child.backward();

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
        //TODO: Does this need a clone? 
        self.gradients_self.clone()
    }

    fn get_gradient_wrt_parent(&self,index: i64) -> 
    Vec<Vec<f64>> {
        self.composer.gradient_wrt_parent(
            &self.parents, index, &self.params)
    }

    fn get_l_size(&self) -> i64 {
        self.gradients_self.len() as i64
    }
}
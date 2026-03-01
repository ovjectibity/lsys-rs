// use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

//NN: 
//Weight matrix multiply & then apply non-linearity
// struct FFN {
    
// }

// impl FFN {
//     fn new() -> Self {
//         Tensor t1 = 

//         FFN {

//         }
//     }
// }

trait ComputationNode {
    fn forward(&mut self);
    fn backward(&mut self);
    fn get_activations(&self) -> Rc<RefCell<Vec<f64>>>;

    fn change_params(&mut self) {

    }

    //Returns [l_size, num_size] Jacobian
    fn get_gradient(&self) -> Rc<RefCell<Vec<f64>>>;

    //Returns [num_size, num_parent_size] Jacobian
    fn get_gradient_wrt_parent(&self,index: i64) -> Option<Vec<f64>>;

    fn get_l_size(&self) -> i64; 
}

trait Optimiser {
    fn step();
}

trait TensorFunction {
    //Returns num_size vec
    fn apply(&self,
        args: &Vec<Tensor>) -> Vec<f64>;
    //Returns [num_size, num_parent_size] Jacobian
    //Uses the parents activations to computie this: 
    fn gradient_wrt_parent(&self,
        args: &Vec<Tensor>,
        index: i64) -> Vec<f64>;
    //Returns [num_size, num_params] Jacobian
    fn gradient_wrt_params(&self,args: &Vec<Tensor>) -> Vec<f64>;
    fn get_num_params(&self) -> i64;
}

struct ReluLayer {

}

impl TensorFunction for ReluLayer {
    fn get_num_params(&self) -> i64 {
        0
    }

    fn apply(&self,
        args: &Vec<Tensor>) -> Vec<f64> {
        assert!(args.len() == 1);
        assert!(args.get(0).expect("").dim == 1);
        let act = &args.get(0).expect("").data;
        act.borrow().iter().map(|e| {
            if *e > 0.0 {
                *e
            } else {
                0.0
            }
        }).collect()
    }

    fn gradient_wrt_parent(&self,
        args: &Vec<Tensor>,
        index: i64) -> Vec<f64> {
        assert!(index == 0);
        assert!(args.len() == 1);
        assert!(args.get(0).expect("").dim == 1);
        let parent_size = args.get(0).expect("").size;
        let act = &args.get(0).expect("").data;
        let mut acc = Vec::new();
        for e in act.borrow().iter().enumerate() {
            let mut grad_part = vec![0.0; parent_size as usize - 1];
            let grad = if *e.1 > 0.0 {
                1.0
            } else {
                0.0
            };
            grad_part.splice(e.0..e.0, [grad]);
            acc.push(grad_part);
        }
        Tensor::flatten(&acc)
    }

    fn gradient_wrt_params(&self,
        args: &Vec<Tensor>) -> Vec<f64> {
        vec![]
    }
}

struct LinearLayer {
    params: Rc<RefCell<Vec<f64>>>
}

impl TensorFunction for LinearLayer {
    fn get_num_params(&self) -> i64 {
        self.params.borrow().len() as i64
    }

    fn apply(&self,
        args: &Vec<Tensor>) -> Vec<f64> {
        //Assert that the args is just 1 dim tensor
        assert!(args.len() == 1);
        assert!(args.get(0).expect("").dim == 1);
        assert!(self.params.borrow().len() as i64 % args.get(0).expect("").size == 0);
        let arg_size = args.get(0).expect("").size as i64;
        let args_activations = 
            Tensor::unflatten(args.get(0).expect("").
                            data.borrow().as_ref(),
                            1);
        
        //Params need to be: [num_size,num_arg_size]
        let params_m = 
            Tensor::unflatten(self.params.borrow().as_ref(), arg_size);
        Tensor::flatten(&Tensor::multiply_matrices(
            &params_m, &args_activations))
    }

    fn gradient_wrt_parent(&self,
        args: &Vec<Tensor>,
        index: i64) -> Vec<f64> {
        assert!(index == 0);
        assert!(args.len() == 1);
        assert!(args.get(0).expect("").dim == 1);
        assert!(self.params.borrow().len() as i64 % args.get(0).expect("").size == 0);
        // let arg_size = args.get(0).expect("").size;
        //Return the weight matrix directly: 
        self.params.borrow().clone()
    }

    //Returns [num_size, num_params] Jacobian
    fn gradient_wrt_params(&self,args: &Vec<Tensor>) -> Vec<f64> {
        assert!(args.len() == 1);
        assert!(args.get(0).expect("").dim == 1);
        assert!(self.params.borrow().len() as i64 % args.get(0).expect("").size == 0);
        let param_len = self.params.borrow().len() as i64;
        let parent_activations = 
            &args.get(0).expect("").data.clone();
        let num_size = param_len / parent_activations.borrow().len() as i64;
        let mut acc = Vec::new();
        for i in 0..num_size {
            let mut grad_params = vec![0.0; param_len as usize - 
                parent_activations.borrow().len() as usize];
            grad_params.splice(i as usize..i as usize, 
                parent_activations.borrow().iter().cloned());
            acc.push(grad_params);
        }
        Tensor::flatten(&acc)
    }
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
    composer: Option<Box<dyn TensorFunction>>,
    //Params numbers are described by the shape
    //This should belong to the function & not to the tensor itself
    // params: Option<Rc<RefCell<Vec<f64>>>>,
    //Stored in the reverse order of the indices represented by shape
    data: Rc<RefCell<Vec<f64>>>,
    //Gradients wrt to self activations & params
    //The gradients are always for a specific tensor known as 
    // the superchild & whose size is l_size
    l_size: i64,
    //[l_size,num_size]
    gradients_self: Rc<RefCell<Vec<f64>>>,
    //[l_size,num_params]
    gradients_params: Rc<RefCell<Vec<f64>>>
}

impl Tensor {
    fn new_direct(shape: Vec<i32>,
        data: Vec<f64>) -> Self {
        let id = 0;
        let size = shape.iter().sum::<i32>() as i64;
        let dim = shape.len();
        Tensor {
            id: id,
            dim: dim as i64,
            size: size,
            shape: shape,
            parents: vec![],
            children: vec![],
            composer: None,
            // params: None,
            data: Rc::new(RefCell::new(data)),
            gradients_self: Rc::new(RefCell::new(vec![])),
            gradients_params: Rc::new(RefCell::new(vec![])),
            l_size: 0
        }
    }

    fn new_composed(shape: Vec<i32>,
        // params: Option<Vec<f64>>,
        parents: Vec<Tensor>,
        children: Vec<Tensor>,
        composer: Box<dyn TensorFunction>) -> Self {
        let id = 0;
        let size = shape.iter().sum::<i32>() as i64;
        let dim = shape.len();
        Tensor {
            id: id,
            dim: dim as i64,
            size: size,
            shape: shape,
            parents: parents,
            children: children,
            composer: Some(composer),
            // params: if let Some(params_v) = params {
            //     Some(Rc::new(RefCell::new(params_v)))
            // } else {
            //     None
            // },
            data: Rc::new(RefCell::new(vec![0.0;size as usize])),
            gradients_self: Rc::new(RefCell::new(vec![])),
            gradients_params: Rc::new(RefCell::new(vec![])),
            l_size: 0
        }
    }

    fn add_matrices(
        m1: &Vec<Vec<f64>>,
        m2: &Vec<Vec<f64>>) -> 
    Vec<Vec<f64>> {
        let mut acc = Vec::new();
        for m1_row in m1.iter().enumerate() {
            let m2_row_v = m2.get(m1_row.0).expect("");
            let mut acc_p = Vec::new();
            for m2_row in m2_row_v.iter().enumerate() {
                acc_p.push(m1_row.1.get(m2_row.0).expect("") + m2_row.1);
            }
            acc.push(acc_p);
        }
        acc
    }

    fn multiply_matrices(m1: &Vec<Vec<f64>>,
        m2: &Vec<Vec<f64>>) -> 
    Vec<Vec<f64>> {
        let a = m1.len();
        let b = m1.get(0).expect("").len();
        assert!(b == m2.len());
        let c = m2.get(0).expect("").len();
        let mut acc = vec![vec![0.0;c];a];
        for m1_row in m1.iter().enumerate() {
            for m1_row_e in m1_row.1.iter().enumerate() {
                for m2_row in m2.iter().enumerate() {
                    acc[m1_row.0][m2_row.0] += 
                        m2_row.1.get(m1_row_e.0).expect("") * m1_row_e.1;
                }
            }
        }
        acc
    }

    fn flatten(
        t1: &Vec<Vec<f64>>) -> 
    Vec<f64> {
        t1.iter().flat_map(|e| {
            e.clone()
        }).collect()
    }

    //Returns [t1.len() / end_dim, end_dim]
    fn unflatten(
        t1: &Vec<f64>,
        end_dim: i64) -> 
    Vec<Vec<f64>> {
        let start_dim = t1.len() / end_dim as usize;
        let mut acc = vec![vec![0.0;end_dim as usize];start_dim];
        for i in 0..start_dim {
            for j in 0..end_dim as usize {
                acc[i][j] = t1[i*end_dim as usize + j];
            }
        }
        acc
    }
}

impl ComputationNode for Tensor {
    fn forward(&mut self) {
        for parent in 
        &mut self.parents {
            parent.forward();
        }
        if let Some(composer) = 
        &self.composer {
            //TODO: Ensure that the shape of vec 
            //returned by the composer is as expected
            let parents = self.parents.as_ref();
            self.data = 
                Rc::new(RefCell::new(composer.apply(
                    parents)));
        }
    }

    fn backward(&mut self) {
        //This function computes gradients with respect to its self activations & params
        //i.e. populates gradients_self & gradients_params: 
        //Computing the gradient_self by using the child's activation gradients 
        //& the gradients of child's activation wrt to parent activations
        let l_size = self.children.get(0).expect("").get_l_size();
        let mut acc = vec![
            vec![0.0];self.size as usize + l_size as usize];
        for child in &mut self.children {
            child.backward();

            //Returns [l_size, num_child_size] array
            let child_grad = 
                child.get_gradient();
            //Returns [num_child_size, num_size] Jacobian
            let child_self_grad = 
                child.get_gradient_wrt_parent(self.id).expect("");
            let child_grad_m = 
                Tensor::unflatten(child_grad.borrow().as_ref(), child.size);
            let child_self_grad_m = 
                Tensor::unflatten(&child_self_grad, self.size);
            let grad_part = Tensor::multiply_matrices(
                &child_grad_m, &child_self_grad_m);
            acc = Tensor::add_matrices(&acc, &grad_part);
        }

        //[l_size,num_size]
        self.gradients_self = Rc::new(RefCell::new(Tensor::flatten(&acc)));
    
        if let Some(composer) = &self.composer {
            //Compute gradients wrt to params now: 
            //[num_size,num_params]
            let x = composer.gradient_wrt_params(&self.parents);
            let grad_params_unflat = Tensor::multiply_matrices(
                &acc,
                &Tensor::unflatten(&x, composer.get_num_params()));
            
            self.gradients_params = Rc::new(RefCell::new(Tensor::flatten(&grad_params_unflat)));
        }
    }

    fn get_activations(&self) -> Rc<RefCell<Vec<f64>>> {
        //TODO: Does this need a clone? 
        //Can we use shallow cloning?
        self.data.clone()
    }

    fn get_gradient(&self) -> Rc<RefCell<Vec<f64>>> {
        //TODO: Does this need a clone? 
        //TODO: Implement this by chopping it up: 
        self.gradients_self.clone()
    }

    fn get_gradient_wrt_parent(&self,index: i64) -> 
    Option<Vec<f64>> {
        if let Some(composer) = 
        &self.composer  {
            Some(composer.gradient_wrt_parent(
                &self.parents, index))
        } else {
            None
        }
    }

    fn get_l_size(&self) -> i64 {
        self.gradients_self.borrow().len() as i64
    }
}
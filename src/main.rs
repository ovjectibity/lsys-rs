pub mod tensor;

use tensor::ComputationNode;

fn main() {
    let ffn = tensor::FFN::new();
    let mut layers = ffn.layers;
    let len = layers.len()-1;
    {
        let last_layer = layers.get_mut(len).expect("");
        last_layer.forward();
    }
    {
        let first_layer = layers.get_mut(0).expect("");
        first_layer.backward();
    }
}

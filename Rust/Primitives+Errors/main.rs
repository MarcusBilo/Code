fn main() {

    // Inferred and immutable
    let default_float1   = 3.0;
    // Annotated and immutable
    let default_float2: f64   = 3.0;
    // Inferred and mutable
    let mut default_float3= 3.0;
    default_float3 = 4.0;
    // Annotated and immutable
    let mut default_float4: f64 = 3.0;
    default_float4 = 4.0;

    // Error Handling using Match statement on Result return
    match divide(10, 0) {
        Ok(result) => println!("Result of division: {}", result),
        Err(e) => println!("Error: {}", e),
    }
}

// &'static str is a reference to a string literal with a static lifetime - error handling standard
fn divide(a: i32, b: i32) -> Result<i32, &'static str> {
    if b == 0 {
        return Err("Division by zero is not allowed");
    }
    Ok(a / b)
}

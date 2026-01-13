# Tensor Logic Implementation

This document describes the Tensor Logic implementation in sbt's logic module, which bridges neural and symbolic AI approaches.

## Overview

Tensor Logic is a framework that unifies neural network operations and symbolic logic programming through tensor equations. This implementation is based on concepts from:

- [Tensor Logic: The Language of AI](https://tensor-logic.org/)
- [Ben Goertzel's work on neural-symbolic integration](https://bengoertzel.substack.com/p/tensor-logic-for-bridging-neural)

## Core Concepts

### 1. Tensor Representations

The implementation provides two types of tensors:

- **DiscreteTensor**: Represents Boolean logic with values restricted to 0.0 (false) or 1.0 (true)
- **ContinuousTensor**: Represents continuous-valued embeddings for neural reasoning

```scala
// Create a discrete tensor for Boolean logic
val discreteTensor = DiscreteTensor(
  shape = Seq(3, 2),
  values = Array(1.0, 0.0, 1.0, 1.0, 0.0, 1.0)
)

// Create a continuous tensor for embeddings
val continuousTensor = ContinuousTensor(
  shape = Seq(3, 4),
  values = Array.fill(12)(0.5)
)
```

### 2. Einstein Summation (einsum)

Einstein summation is the fundamental operation that unifies neural and symbolic computation. It allows expressing:

- Matrix multiplication: `"ij,jk->ik"`
- Inner product: `"i,i->"`
- Matrix-vector multiplication: `"ij,i->j"`

```scala
// Matrix multiplication for relational inference
val relation = DiscreteTensor(Seq(3, 2), Array(1.0, 0.0, 1.0, 1.0, 0.0, 1.0))
val query = DiscreteTensor(Seq(3), Array(1.0, 0.0, 1.0))
val result = TensorLogic.einsum("ij,i->j", relation, query)
```

### 3. Logical Operations

The implementation provides tensor-based logical operations that work for both discrete and continuous tensors:

```scala
// AND operation (product for probabilistic logic)
val andResult = TensorLogic.and(tensorA, tensorB)

// OR operation (probabilistic sum: a + b - a*b)
val orResult = TensorLogic.or(tensorA, tensorB)

// NOT operation (1 - x for fuzzy negation)
val notResult = TensorLogic.not(tensorA)
```

### 4. Embedding Space Reasoning

Convert symbolic atoms to continuous embeddings for neural reasoning:

```scala
val atoms = Set(Atom("parent"), Atom("child"), Atom("grandparent"))
val embeddings = TensorLogic.embed(atoms, embeddingDim = 16)
// Returns Map[Atom, ContinuousTensor] with 16-dimensional embeddings
```

### 5. Neural-Symbolic Bridge

The `TensorLogicBridge` object provides integration between traditional symbolic logic and tensor operations:

```scala
// Convert a formula to tensor representation
val formula = atomA && atomB && (!atomC)
val tensor = TensorLogicBridge.formulaToTensor(formula, embeddings)

// Perform hybrid reasoning combining symbolic and neural approaches
val clauses = Clauses(List(
  Formula.True.proves(Atom("A")),
  Atom("A").proves(Atom("B"))
))
val result = TensorLogicBridge.hybridReason(
  clauses, 
  initialFacts = Set.empty, 
  embeddingDim = 16
)
```

## Advanced Features

### Temperature Scaling

Control the "hardness" of reasoning with temperature scaling:

```scala
// Higher temperature = softer (more fuzzy) reasoning
val softer = TensorLogic.temperatureScale(tensor, temperature = 2.0)

// Lower temperature = harder (more discrete) reasoning
val harder = TensorLogic.temperatureScale(tensor, temperature = 0.5)
```

### Softmax for Probabilistic Reasoning

Convert tensor values to probability distributions:

```scala
val probabilities = TensorLogic.softmax(tensor)
// Ensures values sum to 1.0 and are non-negative
```

### Threshold Operations

Convert continuous tensors to discrete by thresholding:

```scala
val discrete = TensorLogic.threshold(
  continuousTensor, 
  threshold = 0.5
)
```

### Relational Inference

Perform logic programming-style queries using tensor operations:

```scala
// R(x,y) represented as 2D tensor
val relation = DiscreteTensor(Seq(3, 2), relationData)
// Q(x) represented as 1D tensor
val query = DiscreteTensor(Seq(3), queryData)
// Compute answer(y) = R(x,y) * Q(x)
val answer = TensorLogic.relationalInfer(relation, query)
```

## Tensor Operations

### Supported Operations

The `TensorOp` sealed trait defines operations:

- `Einsum(equation: String)`: Einstein summation with equation string
- `Multiply`: Element-wise multiplication
- `Add`: Element-wise addition
- `MatMul`: Matrix multiplication (special case of einsum)
- `And`: Logical AND
- `Or`: Logical OR
- `Not`: Logical negation
- `Threshold(value: Double)`: Apply threshold function
- `TemperatureScaling(temperature: Double)`: Temperature-controlled reasoning

### Tensor Equations

Represent logical rules as tensor equations:

```scala
val input1 = TensorVar("x", Seq(10))
val input2 = TensorVar("y", Seq(10))
val output = TensorVar("z", Seq(10))
val equation = TensorEquation(
  inputs = Seq(input1, input2),
  output = output,
  operation = TensorOp.And
)
```

## Use Cases

### 1. Logic Programming with Tensors

Replace traditional Prolog-style queries with tensor operations for better scalability:

```scala
// Define a relation: parent(X,Y)
val parentRelation = DiscreteTensor(
  shape = Seq(numPeople, numPeople),
  values = parentData
)

// Query: grandparent(X,Z) :- parent(X,Y), parent(Y,Z)
val grandparent = TensorLogic.einsum(
  "ij,jk->ik", 
  parentRelation, 
  parentRelation
)
```

### 2. Fuzzy Logic Reasoning

Use continuous tensors for fuzzy logic:

```scala
val isTall = ContinuousTensor(Seq(3), Array(0.9, 0.5, 0.2))
val isAthletic = ContinuousTensor(Seq(3), Array(0.8, 0.7, 0.3))
// Who is tall AND athletic?
val tallAndAthletic = TensorLogic.and(isTall, isAthletic)
// Result: [0.72, 0.35, 0.06] (probabilistic AND)
```

### 3. Neural-Symbolic Integration

Combine learned embeddings with symbolic reasoning:

```scala
// Create embeddings from learned representations
val embeddings = TensorLogic.embed(atoms, embeddingDim = 32)

// Use symbolic logic to constrain neural reasoning
val formula = (atomA && atomB) || (!atomC)
val constraint = TensorLogicBridge.formulaToTensor(formula, embeddings)

// Apply constraint to neural predictions
val constrained = TensorLogic.and(predictions, constraint)
```

## Implementation Details

### Einstein Summation Implementation

Currently supports:
- `"ij,jk->ik"`: Matrix multiplication
- `"i,i->"`: Inner product (scalar result)
- `"ij,i->j"`: Matrix-vector multiplication

The implementation can be extended to support more complex einsum patterns.

### Probabilistic vs. Fuzzy Logic

- **Probabilistic AND**: Uses product (a * b)
- **Probabilistic OR**: Uses probabilistic sum (a + b - a*b)
- **Fuzzy AND**: Could use minimum (not currently default)
- **Fuzzy OR**: Could use maximum (implemented for discrete tensors)

### Performance Considerations

- Tensor operations use simple array-based implementation
- For production use, consider integration with optimized tensor libraries
- Einstein summation parsing is basic and can be enhanced
- GPU acceleration not yet implemented but operations are designed for it

## Testing

The implementation includes 55 comprehensive unit tests covering:

- Tensor creation and validation
- All logical operations (AND, OR, NOT)
- Einstein summation patterns
- Embedding creation
- Neural-symbolic bridge
- Edge cases and error handling
- Continuous and discrete tensor operations
- Relational inference
- Temperature scaling and softmax

Run tests with:
```bash
./sbt "project logicProj" test
```

## Future Extensions

Potential enhancements:

1. **Extended Einstein Summation**: Support arbitrary einsum patterns
2. **Gradient Computation**: Add autodiff for learning
3. **GPU Acceleration**: Integration with GPU tensor libraries
4. **Graph Neural Networks**: Add GNN-style message passing
5. **Attention Mechanisms**: Implement transformer-style attention
6. **Probabilistic Logic**: Enhanced probabilistic reasoning
7. **Rule Learning**: Learn tensor equations from data
8. **Optimization**: Performance improvements for large tensors

## References

- [Tensor Logic Paper (arXiv:2510.12269)](https://arxiv.org/abs/2510.12269)
- [Tensor Logic Website](https://tensor-logic.org/)
- [Ben Goertzel on Neural-Symbolic AI](https://bengoertzel.substack.com/p/tensor-logic-for-bridging-neural)
- [OpenCog Framework](https://opencog.org/)
- [Neural-Symbolic Integration Survey](https://www.sciencedirect.com/science/article/pii/S2667305325000675)

## License

This implementation is part of sbt and is licensed under Apache License 2.0.

# sbt Util Logic Module

This module provides propositional logic with negation as failure for sbt, along with a new **Tensor Logic** implementation that bridges neural and symbolic AI.

## Components

### 1. Traditional Logic (Logic.scala)

Implements stratified propositional logic with negation as failure. This is used internally by sbt for dependency resolution and configuration logic.

**Key features:**
- Propositional logic with conjunction (AND)
- Negation as failure
- Stratified rule sets (acyclic negation)
- Unique minimal model computation

**Example:**
```scala
val A = Atom("A")
val B = Atom("B")
val clauses = List(
  Formula.True.proves(A),
  A.proves(B)
)
Logic.reduceAll(clauses, Set.empty) // Proves A and B
```

### 2. Tensor Logic (TensorLogic.scala) 🆕

A new implementation that unifies neural network operations and symbolic logic through tensor equations. Based on concepts from [tensor-logic.org](https://tensor-logic.org/) and Ben Goertzel's work on neural-symbolic integration.

**Key features:**
- Einstein summation (einsum) for tensor operations
- Discrete tensors for Boolean logic
- Continuous tensors for embedding space reasoning
- Probabilistic and fuzzy logic operations
- Neural-symbolic bridge for hybrid reasoning
- Temperature-controlled reasoning
- Relational inference (Prolog-style queries with tensors)

**Example:**
```scala
// Boolean logic with tensors
val a = DiscreteTensor(Seq(3), Array(1.0, 0.0, 1.0))
val b = DiscreteTensor(Seq(3), Array(1.0, 1.0, 0.0))
val result = TensorLogic.and(a, b) // [1.0, 0.0, 0.0]

// Probabilistic reasoning
val beliefs = ContinuousTensor(Seq(3), Array(0.8, 0.5, 0.3))
val evidence = ContinuousTensor(Seq(3), Array(0.9, 0.6, 0.4))
val updated = TensorLogic.and(beliefs, evidence) // Probabilistic AND

// Einstein summation for relational queries
val relation = DiscreteTensor(Seq(3, 2), relationData)
val query = DiscreteTensor(Seq(3), queryData)
val answer = TensorLogic.einsum("ij,i->j", relation, query)
```

## Documentation

- **[TENSOR_LOGIC.md](TENSOR_LOGIC.md)**: Comprehensive guide to Tensor Logic
  - Core concepts and architecture
  - API reference
  - Use cases and applications
  - Implementation details

- **[TensorLogicExamples.scala](src/main/scala/sbt/internal/util/logic/TensorLogicExamples.scala)**: 10 runnable examples
  - Boolean logic
  - Probabilistic reasoning
  - Relational inference
  - Einstein summation
  - Embedding operations
  - Neural-symbolic bridge
  - Temperature scaling
  - Complex reasoning chains

## Testing

Run tests:
```bash
./sbt "project logicProj" test
```

Test coverage:
- 6 tests for traditional logic
- 55 tests for tensor logic
- **Total: 61 tests, all passing**

## Use Cases

### Traditional Logic
- sbt plugin dependency resolution
- Configuration constraint solving
- Build task ordering

### Tensor Logic
- Neural-symbolic AI applications
- Fuzzy and probabilistic logic
- Knowledge graph reasoning
- Relational learning
- Explainable AI (combining neural predictions with symbolic rules)
- Smart automation (IoT, robotics)
- Natural language understanding with logic constraints

## References

**Tensor Logic:**
- [Tensor Logic: The Language of AI](https://tensor-logic.org/)
- [arXiv paper (2510.12269)](https://arxiv.org/abs/2510.12269)
- [Ben Goertzel on Neural-Symbolic AI](https://bengoertzel.substack.com/p/tensor-logic-for-bridging-neural)

**Traditional Logic:**
- [Negation as Failure (Wikipedia)](https://en.wikipedia.org/wiki/Negation_as_failure)
- [Stable Model Semantics (Wikipedia)](https://en.wikipedia.org/wiki/Stable_model_semantics)
- [Nonmonotonic Logic (Wikipedia)](https://en.wikipedia.org/wiki/Nonmonotonic_logic)

## License

Apache License 2.0 - See [LICENSE](../../LICENSE)

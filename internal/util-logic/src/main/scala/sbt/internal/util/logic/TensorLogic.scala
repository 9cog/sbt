/*
 * sbt
 * Copyright 2023, Scala center
 * Copyright 2011 - 2022, Lightbend, Inc.
 * Copyright 2008 - 2010, Mark Harrah
 * Licensed under Apache License 2.0 (see LICENSE)
 */

package sbt.internal.util
package logic

/**
 * Tensor Logic: Neural-Symbolic Integration
 *
 * This module implements tensor-based logical reasoning that bridges neural and symbolic AI.
 * Core concepts from Ben Goertzel's Tensor Logic framework:
 * - Logical rules as tensor equations (Einstein summation)
 * - Reasoning in embedding spaces
 * - Integration of continuous and discrete logic
 *
 * References:
 * - https://tensor-logic.org/
 * - https://bengoertzel.substack.com/p/tensor-logic-for-bridging-neural
 */

/**
 * Represents a tensor with arbitrary dimensions.
 * Tensors can represent both discrete (Boolean) and continuous (Real-valued) logic.
 */
sealed trait Tensor {
  def shape: Seq[Int]
  def rank: Int = shape.length
}

/**
 * Discrete tensor for Boolean logic operations.
 * Values are 0.0 (false) or 1.0 (true).
 */
final case class DiscreteTensor(shape: Seq[Int], values: Array[Double]) extends Tensor {
  require(shape.product == values.length, s"Shape product ${shape.product} must match values length ${values.length}")
  require(values.forall(v => v == 0.0 || v == 1.0), "Discrete tensor values must be 0.0 or 1.0")

  override def toString: String = s"DiscreteTensor(shape=${shape.mkString("[", ",", "]")}, size=${values.length})"
}

/**
 * Continuous tensor for embedding-space reasoning.
 * Values are real numbers, typically in a learned embedding space.
 */
final case class ContinuousTensor(shape: Seq[Int], values: Array[Double]) extends Tensor {
  require(shape.product == values.length, s"Shape product ${shape.product} must match values length ${values.length}")

  override def toString: String = s"ContinuousTensor(shape=${shape.mkString("[", ",", "]")}, size=${values.length})"
}

/**
 * A tensor equation represents a logical rule as a tensor operation.
 * This is the fundamental primitive of tensor logic.
 */
final case class TensorEquation(
    inputs: Seq[TensorVar],
    output: TensorVar,
    operation: TensorOp
)

/**
 * Tensor variable with a name and shape specification.
 */
final case class TensorVar(name: String, shape: Seq[Int]) {
  override def toString: String = s"$name[${shape.mkString(",")}]"
}

/**
 * Tensor operations that can be performed.
 */
sealed trait TensorOp

object TensorOp {
  /** Einstein summation - the core operation for tensor logic */
  final case class Einsum(equation: String) extends TensorOp {
    override def toString: String = s"einsum('$equation')"
  }

  /** Element-wise multiplication */
  case object Multiply extends TensorOp

  /** Element-wise addition */
  case object Add extends TensorOp

  /** Matrix multiplication (special case of einsum) */
  case object MatMul extends TensorOp

  /** Logical AND (min for fuzzy logic, product for probabilistic) */
  case object And extends TensorOp

  /** Logical OR (max for fuzzy logic) */
  case object Or extends TensorOp

  /** Negation (1 - x for fuzzy logic) */
  case object Not extends TensorOp

  /** Apply a threshold function (step function for discrete logic) */
  final case class Threshold(value: Double) extends TensorOp

  /** Temperature-controlled reasoning (softmax-style) */
  final case class TemperatureScaling(temperature: Double) extends TensorOp
}

/**
 * Tensor Logic inference engine.
 */
object TensorLogic {

  /**
   * Performs Einstein summation on input tensors.
   * This is the core operation that unifies neural and symbolic computation.
   *
   * Einstein summation allows expressing:
   * - Matrix multiplication: "ij,jk->ik"
   * - Batch matrix multiplication: "bij,bjk->bik"
   * - Inner product: "i,i->"
   * - Outer product: "i,j->ij"
   * - Relational joins: "ij,jk->ik" (same as matrix mul)
   */
  def einsum(equation: String, tensors: Tensor*): Tensor = {
    require(tensors.nonEmpty, "At least one tensor required for einsum")

    // Parse the equation
    val parts = equation.split("->", -1) // Use -1 to preserve trailing empty strings
    require(parts.length == 2, s"Einstein equation must have format 'inputs->output', got: $equation")

    val inputSpec = parts(0).split(",").map(_.trim)
    val outputSpec = parts(1).trim

    require(inputSpec.length == tensors.length, 
      s"Number of input specs (${inputSpec.length}) must match number of tensors (${tensors.length})")

    // For discrete tensors, perform Boolean operations
    val isDiscrete = tensors.forall(_.isInstanceOf[DiscreteTensor])

    if (isDiscrete) {
      performDiscreteEinsum(equation, tensors)
    } else {
      performContinuousEinsum(equation, tensors)
    }
  }

  private def performDiscreteEinsum(equation: String, tensors: Seq[Tensor]): DiscreteTensor = {
    // Simple implementation for basic cases
    // In a full implementation, this would parse and execute arbitrary einsum equations
    equation match {
      case "ij,jk->ik" if tensors.length == 2 =>
        // Matrix multiplication for relational inference
        matrixMultiply(tensors(0).asInstanceOf[DiscreteTensor], tensors(1).asInstanceOf[DiscreteTensor])
      case "i,i->" if tensors.length == 2 =>
        // Inner product (scalar result)
        innerProduct(tensors(0).asInstanceOf[DiscreteTensor], tensors(1).asInstanceOf[DiscreteTensor])
      case "ij,i->j" if tensors.length == 2 =>
        // Matrix-vector multiplication (for relational queries)
        matrixVectorMultiply(tensors(0).asInstanceOf[DiscreteTensor], tensors(1).asInstanceOf[DiscreteTensor])
      case _ =>
        throw new UnsupportedOperationException(s"Einsum equation not yet implemented: $equation")
    }
  }

  private def performContinuousEinsum(equation: String, tensors: Seq[Tensor]): ContinuousTensor = {
    // Similar to discrete but with real-valued operations
    equation match {
      case "ij,jk->ik" if tensors.length == 2 =>
        matrixMultiplyContinuous(
          tensors(0).asInstanceOf[ContinuousTensor],
          tensors(1).asInstanceOf[ContinuousTensor]
        )
      case "i,i->" if tensors.length == 2 =>
        innerProductContinuous(
          tensors(0).asInstanceOf[ContinuousTensor],
          tensors(1).asInstanceOf[ContinuousTensor]
        )
      case "ij,i->j" if tensors.length == 2 =>
        matrixVectorMultiplyContinuous(
          tensors(0).asInstanceOf[ContinuousTensor],
          tensors(1).asInstanceOf[ContinuousTensor]
        )
      case _ =>
        throw new UnsupportedOperationException(s"Einsum equation not yet implemented: $equation")
    }
  }

  /** Matrix multiplication for discrete tensors (Boolean logic) */
  private def matrixMultiply(a: DiscreteTensor, b: DiscreteTensor): DiscreteTensor = {
    require(a.shape.length == 2 && b.shape.length == 2, "Both tensors must be 2D matrices")
    require(a.shape(1) == b.shape(0), s"Inner dimensions must match: ${a.shape(1)} != ${b.shape(0)}")

    val m = a.shape(0)
    val n = b.shape(1)
    val k = a.shape(1)

    val result = Array.fill(m * n)(0.0)

    for {
      i <- 0 until m
      j <- 0 until n
    } {
      var sum = 0.0
      for (p <- 0 until k) {
        val aVal = a.values(i * k + p)
        val bVal = b.values(p * n + j)
        sum += aVal * bVal
      }
      result(i * n + j) = if (sum > 0.0) 1.0 else 0.0 // Threshold to Boolean
    }

    DiscreteTensor(Seq(m, n), result)
  }

  /** Matrix multiplication for continuous tensors */
  private def matrixMultiplyContinuous(a: ContinuousTensor, b: ContinuousTensor): ContinuousTensor = {
    require(a.shape.length == 2 && b.shape.length == 2, "Both tensors must be 2D matrices")
    require(a.shape(1) == b.shape(0), s"Inner dimensions must match: ${a.shape(1)} != ${b.shape(0)}")

    val m = a.shape(0)
    val n = b.shape(1)
    val k = a.shape(1)

    val result = Array.fill(m * n)(0.0)

    for {
      i <- 0 until m
      j <- 0 until n
    } {
      var sum = 0.0
      for (p <- 0 until k) {
        sum += a.values(i * k + p) * b.values(p * n + j)
      }
      result(i * n + j) = sum
    }

    ContinuousTensor(Seq(m, n), result)
  }

  /** Inner product for discrete tensors */
  private def innerProduct(a: DiscreteTensor, b: DiscreteTensor): DiscreteTensor = {
    require(a.shape == b.shape, s"Shapes must match: ${a.shape} != ${b.shape}")
    val sum = (a.values zip b.values).map { case (x, y) => x * y }.sum
    DiscreteTensor(Seq(1), Array(if (sum > 0.0) 1.0 else 0.0))
  }

  /** Inner product for continuous tensors */
  private def innerProductContinuous(a: ContinuousTensor, b: ContinuousTensor): ContinuousTensor = {
    require(a.shape == b.shape, s"Shapes must match: ${a.shape} != ${b.shape}")
    val sum = (a.values zip b.values).map { case (x, y) => x * y }.sum
    ContinuousTensor(Seq(1), Array(sum))
  }

  /** Matrix-vector multiplication for discrete tensors */
  private def matrixVectorMultiply(matrix: DiscreteTensor, vector: DiscreteTensor): DiscreteTensor = {
    require(matrix.shape.length == 2 && vector.shape.length == 1, "First must be 2D matrix, second must be 1D vector")
    require(matrix.shape(0) == vector.shape(0), s"Matrix rows ${matrix.shape(0)} must match vector length ${vector.shape(0)}")

    val m = matrix.shape(0)
    val n = matrix.shape(1)

    val result = Array.fill(n)(0.0)

    for (j <- 0 until n) {
      var sum = 0.0
      for (i <- 0 until m) {
        sum += matrix.values(i * n + j) * vector.values(i)
      }
      result(j) = if (sum > 0.0) 1.0 else 0.0
    }

    DiscreteTensor(Seq(n), result)
  }

  /** Matrix-vector multiplication for continuous tensors */
  private def matrixVectorMultiplyContinuous(matrix: ContinuousTensor, vector: ContinuousTensor): ContinuousTensor = {
    require(matrix.shape.length == 2 && vector.shape.length == 1, "First must be 2D matrix, second must be 1D vector")
    require(matrix.shape(0) == vector.shape(0), s"Matrix rows ${matrix.shape(0)} must match vector length ${vector.shape(0)}")

    val m = matrix.shape(0)
    val n = matrix.shape(1)

    val result = Array.fill(n)(0.0)

    for (j <- 0 until n) {
      var sum = 0.0
      for (i <- 0 until m) {
        sum += matrix.values(i * n + j) * vector.values(i)
      }
      result(j) = sum
    }

    ContinuousTensor(Seq(n), result)
  }

  /**
   * Performs tensor-based logical AND.
   * For discrete: min or product
   * For continuous: product (probabilistic) or min (fuzzy)
   */
  def and(a: Tensor, b: Tensor): Tensor = (a, b) match {
    case (da: DiscreteTensor, db: DiscreteTensor) =>
      require(da.shape == db.shape, "Shapes must match for AND operation")
      val result = (da.values zip db.values).map { case (x, y) => x * y } // Product for Boolean
      DiscreteTensor(da.shape, result)

    case (ca: ContinuousTensor, cb: ContinuousTensor) =>
      require(ca.shape == cb.shape, "Shapes must match for AND operation")
      val result = (ca.values zip cb.values).map { case (x, y) => x * y } // Probabilistic AND
      ContinuousTensor(ca.shape, result)

    case _ =>
      throw new IllegalArgumentException("Cannot mix discrete and continuous tensors in AND")
  }

  /**
   * Performs tensor-based logical OR.
   * For discrete: max
   * For continuous: probabilistic sum (a + b - a*b) or max
   */
  def or(a: Tensor, b: Tensor): Tensor = (a, b) match {
    case (da: DiscreteTensor, db: DiscreteTensor) =>
      require(da.shape == db.shape, "Shapes must match for OR operation")
      val result = (da.values zip db.values).map { case (x, y) => math.max(x, y) }
      DiscreteTensor(da.shape, result)

    case (ca: ContinuousTensor, cb: ContinuousTensor) =>
      require(ca.shape == cb.shape, "Shapes must match for OR operation")
      // Probabilistic sum: P(A or B) = P(A) + P(B) - P(A)*P(B)
      val result = (ca.values zip cb.values).map { case (x, y) => x + y - x * y }
      ContinuousTensor(ca.shape, result)

    case _ =>
      throw new IllegalArgumentException("Cannot mix discrete and continuous tensors in OR")
  }

  /**
   * Performs tensor-based logical NOT.
   * For discrete: 1 - x
   * For continuous: 1 - x (fuzzy negation)
   */
  def not(a: Tensor): Tensor = a match {
    case da: DiscreteTensor =>
      val result = da.values.map(x => 1.0 - x)
      DiscreteTensor(da.shape, result)

    case ca: ContinuousTensor =>
      val result = ca.values.map(x => 1.0 - x)
      ContinuousTensor(ca.shape, result)
  }

  /**
   * Applies temperature scaling for controlled reasoning.
   * Higher temperature makes the reasoning "softer" (more fuzzy).
   * Lower temperature makes it "harder" (more discrete).
   */
  def temperatureScale(tensor: ContinuousTensor, temperature: Double): ContinuousTensor = {
    require(temperature > 0.0, "Temperature must be positive")
    val scaled = tensor.values.map(v => v / temperature)
    ContinuousTensor(tensor.shape, scaled)
  }

  /**
   * Applies softmax for probabilistic reasoning.
   */
  def softmax(tensor: ContinuousTensor): ContinuousTensor = {
    val maxVal = tensor.values.max
    val exps = tensor.values.map(v => math.exp(v - maxVal)) // Subtract max for numerical stability
    val sumExps = exps.sum
    val result = exps.map(_ / sumExps)
    ContinuousTensor(tensor.shape, result)
  }

  /**
   * Converts continuous tensor to discrete by thresholding.
   */
  def threshold(tensor: ContinuousTensor, threshold: Double = 0.5): DiscreteTensor = {
    val result = tensor.values.map(v => if (v >= threshold) 1.0 else 0.0)
    DiscreteTensor(tensor.shape, result)
  }

  /**
   * Creates an embedding space representation for atoms.
   * This enables reasoning in continuous embedding spaces.
   */
  def embed(atoms: Set[Atom], embeddingDim: Int): Map[Atom, ContinuousTensor] = {
    // Simple one-hot encoding as initial embedding
    atoms.zipWithIndex.map { case (atom, idx) =>
      val embedding = Array.fill(embeddingDim)(0.0)
      if (idx < embeddingDim) {
        embedding(idx) = 1.0
      } else {
        // For atoms beyond embedding dimension, use a hash-based approach
        val hash = math.abs(atom.label.hashCode % embeddingDim)
        embedding(hash) = 1.0
      }
      atom -> ContinuousTensor(Seq(embeddingDim), embedding)
    }.toMap
  }

  /**
   * Performs relational inference using tensor operations.
   * This is the key operation for logic programming in tensor form.
   */
  def relationalInfer(
      relation: DiscreteTensor,
      query: DiscreteTensor
  ): DiscreteTensor = {
    // Perform tensor contraction (einsum) for relational queries
    // For example: if relation is R(x,y) and query is Q(x), compute R(x,y) * Q(x) -> answer(y)
    require(
      relation.shape.length == 2 && query.shape.length == 1,
      "Relation must be 2D and query must be 1D"
    )
    require(
      relation.shape(0) == query.shape(0),
      s"Query dimension ${query.shape(0)} must match relation's first dimension ${relation.shape(0)}"
    )

    einsum("ij,i->j", relation, query).asInstanceOf[DiscreteTensor]
  }
}

/**
 * Tensor-based clause representation.
 * Extends the basic clause system with tensor operations.
 */
final case class TensorClause(
    tensorEquation: TensorEquation,
    head: Set[Atom]
) {
  override def toString: String =
    s"TensorClause(${tensorEquation.operation} -> ${head.mkString(", ")})"
}

/**
 * Bridge between traditional logic and tensor logic.
 */
object TensorLogicBridge {

  /**
   * Converts a traditional formula to a tensor representation.
   */
  def formulaToTensor(formula: Formula, atomEmbeddings: Map[Atom, ContinuousTensor]): ContinuousTensor = {
    formula match {
      case atom: Atom =>
        atomEmbeddings.getOrElse(
          atom,
          throw new IllegalArgumentException(s"No embedding found for atom: $atom")
        )

      case Negated(atom) =>
        val embedding = atomEmbeddings.getOrElse(
          atom,
          throw new IllegalArgumentException(s"No embedding found for atom: $atom")
        )
        TensorLogic.not(embedding).asInstanceOf[ContinuousTensor]

      case Formula.And(literals) =>
        // Combine embeddings using AND operation
        literals.foldLeft(
          ContinuousTensor(Seq(atomEmbeddings.values.head.shape.head), Array.fill(atomEmbeddings.values.head.shape.head)(1.0))
        ) { (acc, lit) =>
          val litTensor = literalToTensor(lit, atomEmbeddings)
          TensorLogic.and(acc, litTensor).asInstanceOf[ContinuousTensor]
        }

      case Formula.True =>
        // Return a tensor of all ones
        val dim = atomEmbeddings.values.headOption.map(_.shape.head).getOrElse(1)
        ContinuousTensor(Seq(dim), Array.fill(dim)(1.0))
    }
  }

  private def literalToTensor(literal: Literal, atomEmbeddings: Map[Atom, ContinuousTensor]): ContinuousTensor = {
    literal match {
      case atom: Atom =>
        atomEmbeddings.getOrElse(
          atom,
          throw new IllegalArgumentException(s"No embedding found for atom: $atom")
        )
      case Negated(atom) =>
        val embedding = atomEmbeddings.getOrElse(
          atom,
          throw new IllegalArgumentException(s"No embedding found for atom: $atom")
        )
        TensorLogic.not(embedding).asInstanceOf[ContinuousTensor]
    }
  }

  /**
   * Performs hybrid reasoning combining symbolic and neural approaches.
   */
  def hybridReason(
      clauses: Clauses,
      initialFacts: Set[Literal],
      embeddingDim: Int = 16
  ): Either[Logic.LogicException, (Logic.Matched, Map[Atom, ContinuousTensor])] = {
    // First perform traditional symbolic reasoning
    Logic.reduce(clauses, initialFacts) match {
      case Left(err) => Left(err)
      case Right(matched) =>
        // Then create embeddings for the proven atoms
        val embeddings = TensorLogic.embed(matched.provenSet, embeddingDim)
        Right((matched, embeddings))
    }
  }
}

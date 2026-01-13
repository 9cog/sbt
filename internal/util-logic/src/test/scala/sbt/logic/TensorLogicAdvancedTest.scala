/*
 * sbt
 * Copyright 2023, Scala center
 * Copyright 2011 - 2022, Lightbend, Inc.
 * Copyright 2008 - 2010, Mark Harrah
 * Licensed under Apache License 2.0 (see LICENSE)
 */

package sbt.internal.util
package logic

import org.scalacheck.*
import Prop.secure

/**
 * Additional advanced tests for tensor logic operations.
 * These tests cover edge cases, performance characteristics, and advanced features.
 */
object TensorLogicAdvancedTest extends Properties("TensorLogicAdvanced") {

  // Test: Large tensor operations
  property("Handles large discrete tensors") = secure {
    val size = 1000
    val a = DiscreteTensor(Seq(size), Array.fill(size)(1.0))
    val b = DiscreteTensor(Seq(size), Array.fill(size)(0.0))
    val result = TensorLogic.and(a, b).asInstanceOf[DiscreteTensor]
    result.values.forall(_ == 0.0)
  }

  property("Handles large continuous tensors") = secure {
    val size = 500
    val tensor = ContinuousTensor(Seq(size), Array.fill(size)(0.5))
    val negated = TensorLogic.not(tensor).asInstanceOf[ContinuousTensor]
    negated.values.forall(v => math.abs(v - 0.5) < 0.001)
  }

  // Test: Multiple dimensions
  property("Handles 4D tensors") = secure {
    val shape = Seq(2, 3, 4, 5)
    val size = shape.product
    val tensor = ContinuousTensor(shape, Array.fill(size)(0.1))
    tensor.rank == 4 && tensor.shape == shape && tensor.values.length == size
  }

  property("Handles scalar tensors") = secure {
    val scalar = ContinuousTensor(Seq(1), Array(42.0))
    scalar.rank == 1 && scalar.values(0) == 42.0
  }

  // Test: Chaining many operations
  property("Chains 10 operations correctly") = secure {
    var tensor = ContinuousTensor(Seq(5), Array(0.5, 0.6, 0.7, 0.8, 0.9))
    
    // Chain 10 NOT operations (should return to similar values)
    for (_ <- 0 until 10) {
      tensor = TensorLogic.not(tensor).asInstanceOf[ContinuousTensor]
    }
    
    // After even number of NOTs, should be close to original
    math.abs(tensor.values(0) - 0.5) < 0.001
  }

  // Test: Matrix operations on larger matrices
  property("Multiplies 10x10 matrices") = secure {
    val identity = {
      val values = Array.fill(100)(0.0)
      for (i <- 0 until 10) values(i * 10 + i) = 1.0
      DiscreteTensor(Seq(10, 10), values)
    }
    val data = DiscreteTensor(Seq(10, 10), Array.fill(100)(1.0))
    val result = TensorLogic.einsum("ij,jk->ik", identity, data)
    result.shape == Seq(10, 10)
  }

  // Test: Softmax properties
  property("Softmax produces valid probability distribution") = secure {
    val tensor = ContinuousTensor(Seq(10), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0))
    val probs = TensorLogic.softmax(tensor)
    val sum = probs.values.sum
    val allPositive = probs.values.forall(_ >= 0.0)
    math.abs(sum - 1.0) < 0.001 && allPositive
  }

  property("Softmax is monotonic") = secure {
    val tensor = ContinuousTensor(Seq(3), Array(1.0, 2.0, 3.0))
    val probs = TensorLogic.softmax(tensor)
    // Higher input should give higher probability
    probs.values(0) < probs.values(1) && probs.values(1) < probs.values(2)
  }

  // Test: Temperature scaling effects
  property("High temperature makes distribution more uniform") = secure {
    val tensor = ContinuousTensor(Seq(3), Array(1.0, 2.0, 3.0))
    val highTemp = TensorLogic.temperatureScale(tensor, 10.0)
    val probsHighTemp = TensorLogic.softmax(highTemp)
    
    // With high temperature, probabilities should be more similar
    val maxDiff = probsHighTemp.values.max - probsHighTemp.values.min
    maxDiff < 0.2 // More uniform
  }

  property("Low temperature makes distribution more peaked") = secure {
    val tensor = ContinuousTensor(Seq(3), Array(1.0, 2.0, 3.0))
    val lowTemp = TensorLogic.temperatureScale(tensor, 0.1)
    val probsLowTemp = TensorLogic.softmax(lowTemp)
    
    // With low temperature, highest value should dominate
    probsLowTemp.values(2) > 0.8 // Most probability on highest value
  }

  // Test: Threshold variations
  property("Threshold at 0.0 makes everything 1.0") = secure {
    val tensor = ContinuousTensor(Seq(5), Array(0.1, 0.2, 0.3, 0.4, 0.5))
    val result = TensorLogic.threshold(tensor, 0.0)
    result.values.forall(_ == 1.0)
  }

  property("Threshold at 1.0 makes everything 0.0") = secure {
    val tensor = ContinuousTensor(Seq(5), Array(0.1, 0.2, 0.3, 0.4, 0.5))
    val result = TensorLogic.threshold(tensor, 1.0)
    result.values.forall(_ == 0.0)
  }

  // Test: Relational inference with larger relations
  property("Performs relational inference on 10x10 relation") = secure {
    val relation = DiscreteTensor(Seq(10, 10), Array.fill(100)(1.0))
    val query = DiscreteTensor(Seq(10), Array.fill(10)(1.0))
    val result = TensorLogic.relationalInfer(relation, query)
    result.shape == Seq(10) && result.values.forall(_ == 1.0)
  }

  // Test: Embedding with different dimensions
  property("Creates embeddings with dimension 1") = secure {
    val atoms = Set(Atom("A"), Atom("B"))
    val embeddings = TensorLogic.embed(atoms, 1)
    embeddings.size == 2 && embeddings.values.forall(_.shape == Seq(1))
  }

  property("Creates embeddings with large dimension") = secure {
    val atoms = Set(Atom("A"), Atom("B"), Atom("C"))
    val embeddings = TensorLogic.embed(atoms, 128)
    embeddings.size == 3 && embeddings.values.forall(_.shape == Seq(128))
  }

  property("Creates embeddings for many atoms") = secure {
    val atoms = (1 to 50).map(i => Atom(s"Atom$i")).toSet
    val embeddings = TensorLogic.embed(atoms, 32)
    embeddings.size == 50 && embeddings.values.forall(_.values.sum > 0.0)
  }

  // Test: OR operation edge cases
  property("OR with all zeros gives zeros") = secure {
    val a = DiscreteTensor(Seq(3), Array(0.0, 0.0, 0.0))
    val b = DiscreteTensor(Seq(3), Array(0.0, 0.0, 0.0))
    val result = TensorLogic.or(a, b).asInstanceOf[DiscreteTensor]
    result.values.forall(_ == 0.0)
  }

  property("OR with all ones gives ones") = secure {
    val a = DiscreteTensor(Seq(3), Array(1.0, 1.0, 1.0))
    val b = DiscreteTensor(Seq(3), Array(1.0, 1.0, 1.0))
    val result = TensorLogic.or(a, b).asInstanceOf[DiscreteTensor]
    result.values.forall(_ == 1.0)
  }

  // Test: AND operation edge cases
  property("AND with zeros gives zeros") = secure {
    val a = DiscreteTensor(Seq(3), Array(1.0, 1.0, 1.0))
    val b = DiscreteTensor(Seq(3), Array(0.0, 0.0, 0.0))
    val result = TensorLogic.and(a, b).asInstanceOf[DiscreteTensor]
    result.values.forall(_ == 0.0)
  }

  property("AND with all ones gives ones") = secure {
    val a = DiscreteTensor(Seq(3), Array(1.0, 1.0, 1.0))
    val b = DiscreteTensor(Seq(3), Array(1.0, 1.0, 1.0))
    val result = TensorLogic.and(a, b).asInstanceOf[DiscreteTensor]
    result.values.forall(_ == 1.0)
  }

  // Test: NOT operation properties
  property("Double negation returns to original (discrete)") = secure {
    val tensor = DiscreteTensor(Seq(4), Array(1.0, 0.0, 1.0, 0.0))
    val notNot = TensorLogic.not(TensorLogic.not(tensor).asInstanceOf[DiscreteTensor])
    notNot.asInstanceOf[DiscreteTensor].values.sameElements(tensor.values)
  }

  property("Double negation returns to original (continuous)") = secure {
    val tensor = ContinuousTensor(Seq(3), Array(0.8, 0.5, 0.2))
    val notNot = TensorLogic.not(TensorLogic.not(tensor).asInstanceOf[ContinuousTensor])
    val result = notNot.asInstanceOf[ContinuousTensor]
    (tensor.values zip result.values).forall { case (a, b) => math.abs(a - b) < 0.001 }
  }

  // Test: De Morgan's laws for tensors
  property("De Morgan's law: NOT(A AND B) = NOT(A) OR NOT(B)") = secure {
    val a = ContinuousTensor(Seq(3), Array(0.7, 0.5, 0.3))
    val b = ContinuousTensor(Seq(3), Array(0.6, 0.4, 0.8))
    
    val left = TensorLogic.not(TensorLogic.and(a, b).asInstanceOf[ContinuousTensor])
    val notA = TensorLogic.not(a).asInstanceOf[ContinuousTensor]
    val notB = TensorLogic.not(b).asInstanceOf[ContinuousTensor]
    val right = TensorLogic.or(notA, notB)
    
    // Should be approximately equal for probabilistic logic
    (left.asInstanceOf[ContinuousTensor].values zip right.asInstanceOf[ContinuousTensor].values)
      .forall { case (l, r) => math.abs(l - r) < 0.1 }
  }

  // Test: Matrix-vector operations
  property("Matrix-vector multiplication dimension check") = secure {
    val matrix = ContinuousTensor(Seq(5, 3), Array.fill(15)(1.0))
    val vector = ContinuousTensor(Seq(5), Array.fill(5)(1.0))
    val result = TensorLogic.einsum("ij,i->j", matrix, vector)
    result.shape == Seq(3)
  }

  // Test: Inner product properties
  property("Inner product is commutative") = secure {
    val a = ContinuousTensor(Seq(4), Array(1.0, 2.0, 3.0, 4.0))
    val b = ContinuousTensor(Seq(4), Array(5.0, 6.0, 7.0, 8.0))
    val ab = TensorLogic.einsum("i,i->", a, b).asInstanceOf[ContinuousTensor].values(0)
    val ba = TensorLogic.einsum("i,i->", b, a).asInstanceOf[ContinuousTensor].values(0)
    math.abs(ab - ba) < 0.001
  }

  property("Inner product with zero vector is zero") = secure {
    val a = ContinuousTensor(Seq(5), Array(1.0, 2.0, 3.0, 4.0, 5.0))
    val zero = ContinuousTensor(Seq(5), Array(0.0, 0.0, 0.0, 0.0, 0.0))
    val result = TensorLogic.einsum("i,i->", a, zero).asInstanceOf[ContinuousTensor].values(0)
    math.abs(result) < 0.001
  }

  // Test: Continuous logic properties
  property("Probabilistic AND is commutative") = secure {
    val a = ContinuousTensor(Seq(3), Array(0.7, 0.5, 0.3))
    val b = ContinuousTensor(Seq(3), Array(0.6, 0.8, 0.4))
    val ab = TensorLogic.and(a, b).asInstanceOf[ContinuousTensor]
    val ba = TensorLogic.and(b, a).asInstanceOf[ContinuousTensor]
    (ab.values zip ba.values).forall { case (x, y) => math.abs(x - y) < 0.001 }
  }

  property("Probabilistic OR is commutative") = secure {
    val a = ContinuousTensor(Seq(3), Array(0.7, 0.5, 0.3))
    val b = ContinuousTensor(Seq(3), Array(0.6, 0.8, 0.4))
    val ab = TensorLogic.or(a, b).asInstanceOf[ContinuousTensor]
    val ba = TensorLogic.or(b, a).asInstanceOf[ContinuousTensor]
    (ab.values zip ba.values).forall { case (x, y) => math.abs(x - y) < 0.001 }
  }

  // Test: Boundary values
  property("Handles boundary value 0.0 in continuous tensors") = secure {
    val tensor = ContinuousTensor(Seq(3), Array(0.0, 0.5, 1.0))
    val notTensor = TensorLogic.not(tensor).asInstanceOf[ContinuousTensor]
    notTensor.values.sameElements(Array(1.0, 0.5, 0.0))
  }

  property("Handles boundary value 1.0 in continuous tensors") = secure {
    val tensor = ContinuousTensor(Seq(2), Array(1.0, 1.0))
    val result = TensorLogic.and(tensor, tensor).asInstanceOf[ContinuousTensor]
    result.values.forall(_ == 1.0)
  }

  // Test: Tensor shape validation
  property("Rejects 3D and 2D tensor AND") = secure {
    try {
      val a = ContinuousTensor(Seq(2, 3, 4), Array.fill(24)(0.5))
      val b = ContinuousTensor(Seq(2, 3), Array.fill(6)(0.5))
      TensorLogic.and(a, b)
      false
    } catch {
      case _: IllegalArgumentException => true
      case _ => false
    }
  }
}

/**
 * Integration tests for tensor logic with the existing logic system.
 */
object TensorLogicIntegrationTest extends Properties("TensorLogicIntegration") {

  import TestClauses.*

  // Test: Hybrid reasoning with complex clauses
  property("Hybrid reasoning with multiple rules") = secure {
    val clauses = List(
      Formula.True.proves(A),
      A.proves(B),
      B.proves(C),
      (A && B).proves(D)
    )
    val result = TensorLogicBridge.hybridReason(Clauses(clauses), Set.empty, 10)
    result match {
      case Right((matched, embeddings)) =>
        matched.provenSet == Set(A, B, C, D) && embeddings.size == 4
      case Left(_) => false
    }
  }

  // Test: Hybrid reasoning with negation
  property("Hybrid reasoning handles negation") = secure {
    val clauses = List(
      Formula.True.proves(A),
      (!B).proves(C)
    )
    val result = TensorLogicBridge.hybridReason(Clauses(clauses), Set.empty, 8)
    result match {
      case Right((matched, embeddings)) =>
        matched.provenSet.contains(A) && matched.provenSet.contains(C)
      case Left(_) => false
    }
  }

  // Test: Large embedding dimension
  property("Hybrid reasoning with large embedding dimension") = secure {
    val clauses = List(Formula.True.proves(A))
    val result = TensorLogicBridge.hybridReason(Clauses(clauses), Set.empty, 256)
    result match {
      case Right((_, embeddings)) =>
        embeddings.values.forall(_.shape == Seq(256))
      case Left(_) => false
    }
  }

  // Test: Formula conversion with complex formulas
  property("Converts complex formula with multiple negations") = secure {
    val atoms = Set(A, B, C, D)
    val embeddings = TensorLogic.embed(atoms, 8)
    val formula = A && (!B) && C && (!D)
    val tensor = TensorLogicBridge.formulaToTensor(formula, embeddings)
    tensor.shape == Seq(8)
  }

  // Test: Empty clauses edge case
  property("Handles empty initial facts gracefully") = secure {
    val clauses = List(A.proves(B))
    val result = TensorLogicBridge.hybridReason(Clauses(clauses), Set.empty, 5)
    // Should succeed but prove nothing since A is not in initial facts
    result match {
      case Right((matched, _)) => matched.provenSet.isEmpty
      case Left(_) => false
    }
  }
}

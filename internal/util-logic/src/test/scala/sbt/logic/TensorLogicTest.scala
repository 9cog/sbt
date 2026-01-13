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

object TensorLogicTest extends Properties("TensorLogic") {

  // Test: Basic tensor creation
  property("Creates discrete tensor with correct shape") = secure {
    val tensor = DiscreteTensor(Seq(2, 3), Array(1.0, 0.0, 1.0, 0.0, 1.0, 1.0))
    tensor.shape == Seq(2, 3) && tensor.rank == 2 && tensor.values.length == 6
  }

  property("Creates continuous tensor with correct shape") = secure {
    val tensor = ContinuousTensor(Seq(3, 2), Array(0.5, 0.3, 0.8, 0.2, 0.9, 0.1))
    tensor.shape == Seq(3, 2) && tensor.rank == 2 && tensor.values.length == 6
  }

  property("Rejects discrete tensor with invalid values") = secure {
    try {
      DiscreteTensor(Seq(2), Array(0.5, 0.3))
      false
    } catch {
      case _: IllegalArgumentException => true
      case _ => false
    }
  }

  property("Rejects tensor with mismatched shape and values") = secure {
    try {
      DiscreteTensor(Seq(2, 3), Array(1.0, 0.0, 1.0))
      false
    } catch {
      case _: IllegalArgumentException => true
      case _ => false
    }
  }

  // Test: Tensor AND operation
  property("Performs discrete AND correctly") = secure {
    val a = DiscreteTensor(Seq(3), Array(1.0, 0.0, 1.0))
    val b = DiscreteTensor(Seq(3), Array(1.0, 1.0, 0.0))
    val result = TensorLogic.and(a, b).asInstanceOf[DiscreteTensor]
    result.values.sameElements(Array(1.0, 0.0, 0.0))
  }

  property("Performs continuous AND (probabilistic) correctly") = secure {
    val a = ContinuousTensor(Seq(2), Array(0.8, 0.6))
    val b = ContinuousTensor(Seq(2), Array(0.5, 0.5))
    val result = TensorLogic.and(a, b).asInstanceOf[ContinuousTensor]
    // Probabilistic AND: 0.8*0.5=0.4, 0.6*0.5=0.3
    math.abs(result.values(0) - 0.4) < 0.001 && math.abs(result.values(1) - 0.3) < 0.001
  }

  // Test: Tensor OR operation
  property("Performs discrete OR correctly") = secure {
    val a = DiscreteTensor(Seq(3), Array(1.0, 0.0, 1.0))
    val b = DiscreteTensor(Seq(3), Array(0.0, 1.0, 0.0))
    val result = TensorLogic.or(a, b).asInstanceOf[DiscreteTensor]
    result.values.sameElements(Array(1.0, 1.0, 1.0))
  }

  property("Performs continuous OR (probabilistic) correctly") = secure {
    val a = ContinuousTensor(Seq(2), Array(0.8, 0.6))
    val b = ContinuousTensor(Seq(2), Array(0.5, 0.5))
    val result = TensorLogic.or(a, b).asInstanceOf[ContinuousTensor]
    // Probabilistic OR: 0.8+0.5-0.8*0.5=0.9, 0.6+0.5-0.6*0.5=0.8
    math.abs(result.values(0) - 0.9) < 0.001 && math.abs(result.values(1) - 0.8) < 0.001
  }

  // Test: Tensor NOT operation
  property("Performs discrete NOT correctly") = secure {
    val a = DiscreteTensor(Seq(3), Array(1.0, 0.0, 1.0))
    val result = TensorLogic.not(a).asInstanceOf[DiscreteTensor]
    result.values.sameElements(Array(0.0, 1.0, 0.0))
  }

  property("Performs continuous NOT correctly") = secure {
    val a = ContinuousTensor(Seq(2), Array(0.8, 0.3))
    val result = TensorLogic.not(a).asInstanceOf[ContinuousTensor]
    math.abs(result.values(0) - 0.2) < 0.001 && math.abs(result.values(1) - 0.7) < 0.001
  }

  // Test: Einstein summation - Matrix multiplication
  property("Performs matrix multiplication via einsum") = secure {
    // 2x2 matrix times 2x2 matrix
    val a = DiscreteTensor(Seq(2, 2), Array(1.0, 0.0, 0.0, 1.0)) // Identity matrix
    val b = DiscreteTensor(Seq(2, 2), Array(1.0, 1.0, 0.0, 1.0))
    val result = TensorLogic.einsum("ij,jk->ik", a, b).asInstanceOf[DiscreteTensor]
    result.shape == Seq(2, 2) && result.values.sameElements(Array(1.0, 1.0, 0.0, 1.0))
  }

  property("Performs continuous matrix multiplication via einsum") = secure {
    val a = ContinuousTensor(Seq(2, 2), Array(1.0, 2.0, 3.0, 4.0))
    val b = ContinuousTensor(Seq(2, 2), Array(1.0, 0.0, 0.0, 1.0)) // Identity
    val result = TensorLogic.einsum("ij,jk->ik", a, b).asInstanceOf[ContinuousTensor]
    result.values.sameElements(Array(1.0, 2.0, 3.0, 4.0))
  }

  // Test: Einstein summation - Inner product
  property("Performs inner product via einsum") = secure {
    val a = DiscreteTensor(Seq(3), Array(1.0, 0.0, 1.0))
    val b = DiscreteTensor(Seq(3), Array(1.0, 1.0, 1.0))
    val result = TensorLogic.einsum("i,i->", a, b).asInstanceOf[DiscreteTensor]
    result.shape == Seq(1) && result.values(0) == 1.0 // 1*1 + 0*1 + 1*1 = 2 -> thresholded to 1
  }

  property("Performs continuous inner product via einsum") = secure {
    val a = ContinuousTensor(Seq(3), Array(1.0, 2.0, 3.0))
    val b = ContinuousTensor(Seq(3), Array(1.0, 0.0, 1.0))
    val result = TensorLogic.einsum("i,i->", a, b).asInstanceOf[ContinuousTensor]
    result.shape == Seq(1) && result.values(0) == 4.0 // 1*1 + 2*0 + 3*1 = 4
  }

  // Test: Relational inference
  property("Performs relational inference") = secure {
    // Relation: R(x,y) represented as a 3x2 matrix
    // R(0,0)=1, R(0,1)=0, R(1,0)=1, R(1,1)=1, R(2,0)=0, R(2,1)=1
    val relation = DiscreteTensor(Seq(3, 2), Array(1.0, 0.0, 1.0, 1.0, 0.0, 1.0))
    // Query: Q(x) = [1, 0, 1] - asking about x=0 and x=2
    val query = DiscreteTensor(Seq(3), Array(1.0, 0.0, 1.0))
    val result = TensorLogic.relationalInfer(relation, query)
    // Should get y values where relation holds for x=0 or x=2
    // R(0,0)=1, R(0,1)=0, R(2,0)=0, R(2,1)=1
    // Result: y=0 has 1, y=1 has 1 (since at least one x matches)
    result.shape == Seq(2) && result.values(0) == 1.0 && result.values(1) == 1.0
  }

  // Test: Threshold operation
  property("Applies threshold to continuous tensor") = secure {
    val tensor = ContinuousTensor(Seq(4), Array(0.2, 0.5, 0.7, 0.9))
    val result = TensorLogic.threshold(tensor, 0.5)
    result.values.sameElements(Array(0.0, 1.0, 1.0, 1.0))
  }

  property("Applies custom threshold") = secure {
    val tensor = ContinuousTensor(Seq(3), Array(0.3, 0.7, 0.5))
    val result = TensorLogic.threshold(tensor, 0.6)
    result.values.sameElements(Array(0.0, 1.0, 0.0))
  }

  // Test: Temperature scaling
  property("Applies temperature scaling") = secure {
    val tensor = ContinuousTensor(Seq(3), Array(1.0, 2.0, 3.0))
    val result = TensorLogic.temperatureScale(tensor, 2.0)
    result.values.sameElements(Array(0.5, 1.0, 1.5))
  }

  // Test: Softmax
  property("Applies softmax correctly") = secure {
    val tensor = ContinuousTensor(Seq(3), Array(1.0, 2.0, 3.0))
    val result = TensorLogic.softmax(tensor)
    // Check that sum is approximately 1.0
    val sum = result.values.sum
    math.abs(sum - 1.0) < 0.001 && result.values.forall(_ >= 0.0)
  }

  // Test: Embedding creation
  property("Creates embeddings for atoms") = secure {
    val atoms = Set(Atom("A"), Atom("B"), Atom("C"))
    val embeddings = TensorLogic.embed(atoms, 5)
    embeddings.size == 3 && embeddings.values.forall(_.shape == Seq(5))
  }

  property("Embeddings are valid tensors") = secure {
    val atoms = Set(Atom("X"), Atom("Y"))
    val embeddings = TensorLogic.embed(atoms, 4)
    embeddings.values.forall { tensor =>
      tensor.shape == Seq(4) && tensor.values.length == 4
    }
  }

  // Test: Tensor variables and equations
  property("Creates tensor variable") = secure {
    val tvar = TensorVar("input", Seq(10, 20))
    tvar.name == "input" && tvar.shape == Seq(10, 20)
  }

  property("Creates tensor equation") = secure {
    val input1 = TensorVar("x", Seq(3))
    val input2 = TensorVar("y", Seq(3))
    val output = TensorVar("z", Seq(3))
    val eq = TensorEquation(Seq(input1, input2), output, TensorOp.Add)
    eq.inputs.length == 2 && eq.output == output
  }

  // Test: Tensor operations enum
  property("Creates einsum operation") = secure {
    val op = TensorOp.Einsum("ij,jk->ik")
    op.equation == "ij,jk->ik"
  }

  property("Creates threshold operation") = secure {
    val op = TensorOp.Threshold(0.5)
    op.value == 0.5
  }

  property("Creates temperature scaling operation") = secure {
    val op = TensorOp.TemperatureScaling(2.0)
    op.temperature == 2.0
  }

  // Test: Complex logical operations
  property("Combines AND and NOT operations") = secure {
    val a = DiscreteTensor(Seq(2), Array(1.0, 0.0))
    val b = DiscreteTensor(Seq(2), Array(1.0, 1.0))
    val notA = TensorLogic.not(a).asInstanceOf[DiscreteTensor]
    val result = TensorLogic.and(notA, b).asInstanceOf[DiscreteTensor]
    result.values.sameElements(Array(0.0, 1.0))
  }

  property("Combines OR and AND operations") = secure {
    val a = DiscreteTensor(Seq(2), Array(1.0, 0.0))
    val b = DiscreteTensor(Seq(2), Array(0.0, 1.0))
    val c = DiscreteTensor(Seq(2), Array(1.0, 1.0))
    val orResult = TensorLogic.or(a, b).asInstanceOf[DiscreteTensor]
    val finalResult = TensorLogic.and(orResult, c).asInstanceOf[DiscreteTensor]
    finalResult.values.sameElements(Array(1.0, 1.0))
  }

  // Test: Continuous logic operations
  property("Performs fuzzy AND with continuous values") = secure {
    val a = ContinuousTensor(Seq(3), Array(0.9, 0.5, 0.1))
    val b = ContinuousTensor(Seq(3), Array(0.8, 0.6, 0.9))
    val result = TensorLogic.and(a, b).asInstanceOf[ContinuousTensor]
    // Check probabilistic AND
    math.abs(result.values(0) - 0.72) < 0.01 && // 0.9 * 0.8
      math.abs(result.values(1) - 0.3) < 0.01 && // 0.5 * 0.6
      math.abs(result.values(2) - 0.09) < 0.01 // 0.1 * 0.9
  }

  property("Performs fuzzy OR with continuous values") = secure {
    val a = ContinuousTensor(Seq(2), Array(0.3, 0.7))
    val b = ContinuousTensor(Seq(2), Array(0.4, 0.2))
    val result = TensorLogic.or(a, b).asInstanceOf[ContinuousTensor]
    // P(A or B) = P(A) + P(B) - P(A)*P(B)
    // 0.3 + 0.4 - 0.3*0.4 = 0.58
    // 0.7 + 0.2 - 0.7*0.2 = 0.76
    math.abs(result.values(0) - 0.58) < 0.01 &&
      math.abs(result.values(1) - 0.76) < 0.01
  }

  // Test: Multi-dimensional tensors
  property("Handles 3D tensors") = secure {
    val tensor = ContinuousTensor(Seq(2, 3, 4), Array.fill(24)(0.5))
    tensor.rank == 3 && tensor.shape.product == 24
  }

  property("Handles 1D tensors (vectors)") = secure {
    val tensor = DiscreteTensor(Seq(5), Array(1.0, 0.0, 1.0, 1.0, 0.0))
    tensor.rank == 1 && tensor.shape == Seq(5)
  }

  // Test: Edge cases
  property("Handles empty shape gracefully") = secure {
    try {
      DiscreteTensor(Seq(), Array())
      false // Should require at least one dimension
    } catch {
      case _: IllegalArgumentException => true
      case _ => false
    }
  }

  property("Rejects negative temperature") = secure {
    try {
      val tensor = ContinuousTensor(Seq(2), Array(1.0, 2.0))
      TensorLogic.temperatureScale(tensor, -1.0)
      false
    } catch {
      case _: IllegalArgumentException => true
      case _ => false
    }
  }

  property("Rejects mismatched shapes in AND") = secure {
    try {
      val a = DiscreteTensor(Seq(2), Array(1.0, 0.0))
      val b = DiscreteTensor(Seq(3), Array(1.0, 0.0, 1.0))
      TensorLogic.and(a, b)
      false
    } catch {
      case _: IllegalArgumentException => true
      case _ => false
    }
  }

  property("Rejects mismatched shapes in OR") = secure {
    try {
      val a = ContinuousTensor(Seq(2, 2), Array(0.5, 0.5, 0.5, 0.5))
      val b = ContinuousTensor(Seq(2, 3), Array.fill(6)(0.5))
      TensorLogic.or(a, b)
      false
    } catch {
      case _: IllegalArgumentException => true
      case _ => false
    }
  }

  // Test: Relational inference edge cases
  property("Rejects invalid relational inference dimensions") = secure {
    try {
      val relation = DiscreteTensor(Seq(3, 2), Array.fill(6)(1.0))
      val query = DiscreteTensor(Seq(4), Array.fill(4)(1.0)) // Wrong dimension
      TensorLogic.relationalInfer(relation, query)
      false
    } catch {
      case _: IllegalArgumentException => true
      case _ => false
    }
  }

  // Test: Chaining operations
  property("Chains multiple operations correctly") = secure {
    val a = ContinuousTensor(Seq(3), Array(0.8, 0.6, 0.4))
    val b = ContinuousTensor(Seq(3), Array(0.7, 0.5, 0.3))
    // (a AND b) OR NOT(a)
    val andResult = TensorLogic.and(a, b).asInstanceOf[ContinuousTensor]
    val notA = TensorLogic.not(a).asInstanceOf[ContinuousTensor]
    val finalResult = TensorLogic.or(andResult, notA).asInstanceOf[ContinuousTensor]
    finalResult.shape == Seq(3) && finalResult.values.length == 3
  }
}

object TensorLogicBridgeTest extends Properties("TensorLogicBridge") {

  import TestClauses.*

  // Test: Formula to tensor conversion
  property("Converts atom to tensor") = secure {
    val atomA = Atom("A")
    val embeddings = TensorLogic.embed(Set(atomA), 4)
    val tensor = TensorLogicBridge.formulaToTensor(atomA, embeddings)
    tensor.shape == Seq(4)
  }

  property("Converts negated atom to tensor") = secure {
    val atomA = Atom("A")
    val embeddings = TensorLogic.embed(Set(atomA), 4)
    val negatedA = Negated(atomA)
    val tensor = TensorLogicBridge.formulaToTensor(negatedA, embeddings)
    tensor.shape == Seq(4)
  }

  property("Converts AND formula to tensor") = secure {
    val atomA = Atom("A")
    val atomB = Atom("B")
    val embeddings = TensorLogic.embed(Set(atomA, atomB), 5)
    val formula = atomA && atomB
    val tensor = TensorLogicBridge.formulaToTensor(formula, embeddings)
    tensor.shape == Seq(5)
  }

  property("Converts True formula to tensor") = secure {
    val atomA = Atom("A")
    val embeddings = TensorLogic.embed(Set(atomA), 3)
    val tensor = TensorLogicBridge.formulaToTensor(Formula.True, embeddings)
    tensor.shape == Seq(3) && tensor.values.forall(_ == 1.0)
  }

  // Test: Hybrid reasoning
  property("Performs hybrid reasoning") = secure {
    val cs = Formula.True.proves(A) :: Nil
    val result = TensorLogicBridge.hybridReason(Clauses(cs), Set.empty, 8)
    result match {
      case Right((matched, embeddings)) =>
        matched.provenSet.contains(A) && embeddings.contains(A)
      case Left(_) => false
    }
  }

  property("Hybrid reasoning creates embeddings for all proven atoms") = secure {
    val cs =
      Formula.True.proves(A) ::
        Formula.True.proves(B) ::
        (A && B).proves(C) ::
        Nil
    val result = TensorLogicBridge.hybridReason(Clauses(cs), Set.empty, 10)
    result match {
      case Right((matched, embeddings)) =>
        matched.provenSet == Set(A, B, C) &&
          embeddings.keySet == Set(A, B, C) &&
          embeddings.values.forall(_.shape == Seq(10))
      case Left(_) => false
    }
  }

  property("Hybrid reasoning propagates logic errors") = secure {
    // Test that contradictory initial facts (A and !A) cause an error
    val cs = Formula.True.proves(B) :: Nil
    val init = Set[Literal](Atom("A"), Negated(Atom("A")))
    val result = TensorLogicBridge.hybridReason(Clauses(cs), init, 5)
    // This should fail because A and !A are contradictory
    result.isLeft
  }

  // Test: Complex formula conversions
  property("Converts complex AND formula with negation") = secure {
    val atomA = Atom("A")
    val atomB = Atom("B")
    val atomC = Atom("C")
    val embeddings = TensorLogic.embed(Set(atomA, atomB, atomC), 6)
    val formula = atomA && (!atomB) && atomC
    val tensor = TensorLogicBridge.formulaToTensor(formula, embeddings)
    tensor.shape == Seq(6)
  }

  // Test: Missing embeddings
  property("Throws error for missing atom embedding") = secure {
    try {
      val atomA = Atom("A")
      val atomB = Atom("B")
      val embeddings = TensorLogic.embed(Set(atomA), 4)
      TensorLogicBridge.formulaToTensor(atomB, embeddings)
      false
    } catch {
      case _: IllegalArgumentException => true
      case _ => false
    }
  }
}

object TensorClauseTest extends Properties("TensorClause") {

  property("Creates tensor clause") = secure {
    val input = TensorVar("x", Seq(3))
    val output = TensorVar("y", Seq(3))
    val equation = TensorEquation(Seq(input), output, TensorOp.Not)
    val clause = TensorClause(equation, Set(Atom("result")))
    clause.head.contains(Atom("result"))
  }

  property("Tensor clause has meaningful toString") = secure {
    val input = TensorVar("x", Seq(2))
    val output = TensorVar("y", Seq(2))
    val equation = TensorEquation(Seq(input), output, TensorOp.And)
    val clause = TensorClause(equation, Set(Atom("A"), Atom("B")))
    val str = clause.toString
    str.contains("TensorClause") && str.contains("And")
  }
}

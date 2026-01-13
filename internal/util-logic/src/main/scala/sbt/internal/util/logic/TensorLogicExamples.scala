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
 * Examples demonstrating Tensor Logic usage.
 * These examples show how to use the tensor logic implementation for various
 * neural-symbolic AI tasks.
 */
object TensorLogicExamples {

  /**
   * Example 1: Basic Boolean Logic with Tensors
   * Demonstrates discrete tensor operations for traditional logic.
   */
  def booleanLogicExample(): Unit = {
    println("=== Example 1: Boolean Logic with Tensors ===")

    // Create Boolean tensors
    val truthValues = DiscreteTensor(Seq(4), Array(1.0, 0.0, 1.0, 1.0))
    val conditions = DiscreteTensor(Seq(4), Array(1.0, 1.0, 0.0, 1.0))

    // Logical AND
    val andResult = TensorLogic.and(truthValues, conditions)
    println(s"AND: ${andResult.asInstanceOf[DiscreteTensor].values.mkString(", ")}")
    // Output: 1.0, 0.0, 0.0, 1.0

    // Logical OR
    val orResult = TensorLogic.or(truthValues, conditions)
    println(s"OR: ${orResult.asInstanceOf[DiscreteTensor].values.mkString(", ")}")
    // Output: 1.0, 1.0, 1.0, 1.0

    // Logical NOT
    val notResult = TensorLogic.not(truthValues)
    println(s"NOT: ${notResult.asInstanceOf[DiscreteTensor].values.mkString(", ")}")
    // Output: 0.0, 1.0, 0.0, 0.0
  }

  /**
   * Example 2: Probabilistic Reasoning
   * Demonstrates continuous tensor operations for fuzzy/probabilistic logic.
   */
  def probabilisticReasoningExample(): Unit = {
    println("\n=== Example 2: Probabilistic Reasoning ===")

    // Probability that it's raining in different cities
    val isRaining = ContinuousTensor(Seq(3), Array(0.8, 0.3, 0.6))

    // Probability that you'll need an umbrella
    val needUmbrella = ContinuousTensor(Seq(3), Array(0.9, 0.5, 0.7))

    // Probability of raining AND needing umbrella (probabilistic AND)
    val rainAndUmbrella = TensorLogic.and(isRaining, needUmbrella)
    println(s"P(rain AND umbrella): ${rainAndUmbrella.asInstanceOf[ContinuousTensor].values.mkString(", ")}")
    // Output: 0.72, 0.15, 0.42

    // Probability of raining OR needing umbrella (probabilistic OR)
    val rainOrUmbrella = TensorLogic.or(isRaining, needUmbrella)
    println(s"P(rain OR umbrella): ${rainOrUmbrella.asInstanceOf[ContinuousTensor].values.mkString(", ")}")
    // Output: 0.98, 0.65, 0.88
  }

  /**
   * Example 3: Relational Inference (Logic Programming)
   * Demonstrates using tensors for Prolog-style queries.
   */
  def relationalInferenceExample(): Unit = {
    println("\n=== Example 3: Relational Inference ===")

    // Define a "parent" relation: parent(X, Y) means X is parent of Y
    // Rows: [Alice, Bob, Charlie]
    // Cols: [Diana, Eve]
    // Alice is parent of Diana (1,0), Bob is parent of both (1,1), Charlie is parent of Eve (0,1)
    val parentRelation = DiscreteTensor(
      Seq(3, 2),
      Array(1.0, 0.0, 1.0, 1.0, 0.0, 1.0)
    )

    // Query: Which children does Alice have? (query for row 0)
    val aliceQuery = DiscreteTensor(Seq(3), Array(1.0, 0.0, 0.0))
    val alicesChildren = TensorLogic.relationalInfer(parentRelation, aliceQuery)
    println(s"Alice's children: ${alicesChildren.values.mkString(", ")}")
    // Output: 1.0, 0.0 (Diana but not Eve)

    // Compute grandparent relation: grandparent(X,Z) :- parent(X,Y), parent(Y,Z)
    // If we had another generation, we'd use: einsum("ij,jk->ik", parent, parent)
    println("Grandparent computation would use matrix multiplication of parent relation")
  }

  /**
   * Example 4: Einstein Summation for Matrix Operations
   * Demonstrates the power of einsum for tensor computations.
   */
  def einsumExample(): Unit = {
    println("\n=== Example 4: Einstein Summation ===")

    // Matrix multiplication
    val matrixA = ContinuousTensor(Seq(2, 3), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    val matrixB = ContinuousTensor(Seq(3, 2), Array(1.0, 0.0, 0.0, 1.0, 1.0, 1.0))
    val product = TensorLogic.einsum("ij,jk->ik", matrixA, matrixB)
    println(s"Matrix multiplication result: ${product.asInstanceOf[ContinuousTensor].values.mkString(", ")}")

    // Inner product
    val vectorX = ContinuousTensor(Seq(4), Array(1.0, 2.0, 3.0, 4.0))
    val vectorY = ContinuousTensor(Seq(4), Array(1.0, 0.0, 1.0, 0.0))
    val dot = TensorLogic.einsum("i,i->", vectorX, vectorY)
    println(s"Inner product: ${dot.asInstanceOf[ContinuousTensor].values(0)}")
    // Output: 4.0 (1*1 + 2*0 + 3*1 + 4*0)
  }

  /**
   * Example 5: Embedding Space Reasoning
   * Demonstrates creating and using embeddings for symbolic atoms.
   */
  def embeddingExample(): Unit = {
    println("\n=== Example 5: Embedding Space Reasoning ===")

    // Create embeddings for atoms
    val atoms = Set(Atom("dog"), Atom("cat"), Atom("bird"))
    val embeddings = TensorLogic.embed(atoms, embeddingDim = 8)

    println(s"Created ${embeddings.size} embeddings with dimension 8")
    embeddings.foreach { case (atom, tensor) =>
      println(s"${atom.label}: ${tensor.values.take(4).mkString(", ")}...")
    }

    // Perform operations in embedding space
    val dogEmbed = embeddings(Atom("dog"))
    val catEmbed = embeddings(Atom("cat"))

    // Compute similarity (inner product)
    val similarity = TensorLogic.einsum("i,i->", dogEmbed, catEmbed)
    println(s"\nSimilarity between dog and cat: ${similarity.asInstanceOf[ContinuousTensor].values(0)}")
  }

  /**
   * Example 6: Neural-Symbolic Bridge
   * Demonstrates hybrid reasoning combining symbolic logic and neural embeddings.
   */
  def neuralSymbolicBridgeExample(): Unit = {
    println("\n=== Example 6: Neural-Symbolic Bridge ===")

    // Define symbolic rules
    val A = Atom("A")
    val B = Atom("B")
    val C = Atom("C")

    val clauses = List(
      Formula.True.proves(A), // A is always true
      A.proves(B), // If A then B
      (A && B).proves(C) // If A and B then C
    )

    // Perform hybrid reasoning
    val result = TensorLogicBridge.hybridReason(
      Clauses(clauses),
      initialFacts = Set.empty,
      embeddingDim = 16
    )

    result match {
      case Right((matched, embeddings)) =>
        println(s"Proven atoms: ${matched.provenSet.map(_.label).mkString(", ")}")
        println(s"Created embeddings for ${embeddings.size} atoms")
        println("Embeddings can now be used for neural reasoning tasks")
      case Left(error) =>
        println(s"Logic error: $error")
    }
  }

  /**
   * Example 7: Temperature-Controlled Reasoning
   * Demonstrates soft vs. hard reasoning with temperature scaling.
   */
  def temperatureControlExample(): Unit = {
    println("\n=== Example 7: Temperature-Controlled Reasoning ===")

    // Start with some fuzzy beliefs
    val beliefs = ContinuousTensor(Seq(5), Array(0.9, 0.7, 0.5, 0.3, 0.1))

    // High temperature = softer, more fuzzy
    val soft = TensorLogic.temperatureScale(beliefs, temperature = 2.0)
    println(s"High temp (soft): ${soft.values.mkString(", ")}")
    // Values become less extreme

    // Low temperature = harder, more discrete
    val hard = TensorLogic.temperatureScale(beliefs, temperature = 0.5)
    println(s"Low temp (hard): ${hard.values.mkString(", ")}")
    // Values become more extreme

    // Convert to discrete with threshold
    val discrete = TensorLogic.threshold(beliefs, threshold = 0.5)
    println(s"Thresholded: ${discrete.values.mkString(", ")}")
    // Output: 1.0, 1.0, 1.0, 0.0, 0.0
  }

  /**
   * Example 8: Complex Logical Reasoning
   * Demonstrates chaining multiple tensor operations for complex reasoning.
   */
  def complexReasoningExample(): Unit = {
    println("\n=== Example 8: Complex Logical Reasoning ===")

    // Scenario: Smart home automation
    // Sensor readings (probabilities)
    val motionDetected = ContinuousTensor(Seq(3), Array(0.9, 0.2, 0.7)) // Living room, bedroom, kitchen
    val lightLevel = ContinuousTensor(Seq(3), Array(0.1, 0.05, 0.3)) // Low light in all rooms
    val timeOfDay = ContinuousTensor(Seq(3), Array(0.8, 0.8, 0.8)) // Evening

    // Logic: Turn on lights if (motion detected AND low light) OR it's evening
    val motionAndLowLight = TensorLogic.and(motionDetected, 
      TensorLogic.not(lightLevel).asInstanceOf[ContinuousTensor])
    val shouldTurnOnLights = TensorLogic.or(motionAndLowLight.asInstanceOf[ContinuousTensor], timeOfDay)

    println(s"Motion detected: ${motionDetected.values.mkString(", ")}")
    println(s"Light level: ${lightLevel.values.mkString(", ")}")
    println(s"Time of day (evening): ${timeOfDay.values.mkString(", ")}")
    println(s"Should turn on lights: ${shouldTurnOnLights.asInstanceOf[ContinuousTensor].values.mkString(", ")}")

    // Apply threshold for final decision
    val decision = TensorLogic.threshold(shouldTurnOnLights.asInstanceOf[ContinuousTensor], 0.7)
    println(s"Final decision (binary): ${decision.values.mkString(", ")}")
  }

  /**
   * Example 9: Softmax for Probabilistic Interpretation
   * Demonstrates converting tensor values to probability distributions.
   */
  def softmaxExample(): Unit = {
    println("\n=== Example 9: Softmax for Probabilities ===")

    // Logits from a neural network or scoring function
    val logits = ContinuousTensor(Seq(4), Array(2.0, 1.0, 0.5, 0.1))

    // Convert to probabilities
    val probabilities = TensorLogic.softmax(logits)

    println(s"Logits: ${logits.values.mkString(", ")}")
    println(s"Probabilities: ${probabilities.values.map(v => f"$v%.4f").mkString(", ")}")
    println(s"Sum of probabilities: ${probabilities.values.sum}")
    // Sum should be approximately 1.0
  }

  /**
   * Example 10: Building a Tensor Equation
   * Demonstrates creating structured tensor operations.
   */
  def tensorEquationExample(): Unit = {
    println("\n=== Example 10: Tensor Equations ===")

    // Define tensor variables
    val input1 = TensorVar("sensor_data", Seq(10))
    val input2 = TensorVar("prior_beliefs", Seq(10))
    val output = TensorVar("updated_beliefs", Seq(10))

    // Create a tensor equation representing a logical rule
    val updateRule = TensorEquation(
      inputs = Seq(input1, input2),
      output = output,
      operation = TensorOp.And
    )

    println(s"Created tensor equation:")
    println(s"  Inputs: ${updateRule.inputs.map(_.name).mkString(", ")}")
    println(s"  Output: ${updateRule.output.name}")
    println(s"  Operation: ${updateRule.operation}")

    // Create a tensor clause (symbolic rule with tensor operations)
    val clause = TensorClause(
      tensorEquation = updateRule,
      head = Set(Atom("belief_updated"))
    )

    println(s"\nTensor clause: $clause")
  }

  /**
   * Run all examples
   */
  def main(args: Array[String]): Unit = {
    println("Tensor Logic Examples")
    println("=" * 50)

    booleanLogicExample()
    probabilisticReasoningExample()
    relationalInferenceExample()
    einsumExample()
    embeddingExample()
    neuralSymbolicBridgeExample()
    temperatureControlExample()
    complexReasoningExample()
    softmaxExample()
    tensorEquationExample()

    println("\n" + "=" * 50)
    println("All examples completed!")
  }
}

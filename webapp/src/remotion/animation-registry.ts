import type { AnimationMeta } from "./types";
import { DIMENSIONS, FPS } from "./theme";
import MatrixMultiplication from "./modules/mod01/MatrixMultiplication";
import MemoryComparison from "./modules/mod01/MemoryComparison";
import AttentionMechanism from "./modules/mod02a/AttentionMechanism";
import ScaledDotProduct from "./modules/mod02a/ScaledDotProduct";
import MultiHeadAttention from "./modules/mod02b/MultiHeadAttention";
import FeedForwardNetwork from "./modules/mod02b/FeedForwardNetwork";
import TransformerBlockFlow from "./modules/mod02c/TransformerBlockFlow";
import ParameterDistribution from "./modules/mod02c/ParameterDistribution";
import QuantizationVisualization from "./modules/mod03/QuantizationVisualization";
import STEVisualization from "./modules/mod03/STEVisualization";
import TimelineAnimation from "./modules/mod04/TimelineAnimation";
import BinaryVsTernary from "./modules/mod04/BinaryVsTernary";
import BitLinearLayerFlow from "./modules/mod05/BitLinearLayerFlow";
import WeightQuantizationSteps from "./modules/mod05/WeightQuantizationSteps";
import BitLinearVsLinear from "./modules/mod06/BitLinearVsLinear";
import TrainingLoop from "./modules/mod06/TrainingLoop";
import ScalingCurve from "./modules/mod07/ScalingCurve";
import LambdaScheduling from "./modules/mod07/LambdaScheduling";
import WeightPacking from "./modules/mod08/WeightPacking";
import EnergyComparison from "./modules/mod08/EnergyComparison";

function meta(
  id: string,
  title: string,
  component: AnimationMeta["component"],
  durationSeconds: number,
): AnimationMeta {
  return {
    id,
    title,
    component,
    durationInFrames: durationSeconds * FPS,
    fps: FPS,
    width: DIMENSIONS.width,
    height: DIMENSIONS.height,
  };
}

const registry: Record<string, AnimationMeta[]> = {
  "mod-01": [
    meta("mod01-matrix", "Matrix Multiplication with Ternary Weights", MatrixMultiplication, 12),
    meta("mod01-memory", "Memory Comparison: FP32 to Ternary", MemoryComparison, 8),
  ],
  "mod-02a": [
    meta("mod02a-attention", "Attention Q/K/V Flow", AttentionMechanism, 15),
    meta("mod02a-dotproduct", "Scaled Dot-Product Step-by-Step", ScaledDotProduct, 12),
  ],
  "mod-02b": [
    meta("mod02b-multihead", "Multi-Head Attention with GQA", MultiHeadAttention, 10),
    meta("mod02b-ffn", "Feed-Forward Network Pipeline", FeedForwardNetwork, 9),
  ],
  "mod-02c": [
    meta("mod02c-transformer", "Decoder Block Data Flow", TransformerBlockFlow, 10),
    meta("mod02c-params", "Parameter Distribution in BitNet", ParameterDistribution, 9),
  ],
  "mod-03": [
    meta("mod03-quantization", "Quantization: Snapping to Grid Points", QuantizationVisualization, 12),
    meta("mod03-ste", "Straight-Through Estimator", STEVisualization, 10),
  ],
  "mod-04": [
    meta("mod04-timeline", "Binary Neural Networks Timeline: 2015-2025", TimelineAnimation, 12),
    meta("mod04-binaryvternary", "Binary vs Ternary Weight Expressiveness", BinaryVsTernary, 8),
  ],
  "mod-05": [
    meta("mod05-bitlinear", "BitLinear Layer Pipeline", BitLinearLayerFlow, 12),
    meta("mod05-weightquant", "Weight Quantization Steps", WeightQuantizationSteps, 10),
  ],
  "mod-06": [
    meta("mod06-comparison", "nn.Linear vs BitLinear", BitLinearVsLinear, 10),
    meta("mod06-training", "Training Loop Cycle", TrainingLoop, 12),
  ],
  "mod-07": [
    meta("mod07-scaling", "Scaling Curve: BitNet vs FP16", ScalingCurve, 10),
    meta("mod07-lambda", "Lambda Scheduling During Fine-Tuning", LambdaScheduling, 10),
  ],
  "mod-08": [
    meta("mod08-packing", "Weight Packing: Ternary to 2-Bit Encoding", WeightPacking, 10),
    meta("mod08-energy", "Energy Comparison: 71x Reduction", EnergyComparison, 8),
  ],
};

export function getAnimationsForModule(moduleId: string): AnimationMeta[] {
  return registry[moduleId] ?? [];
}

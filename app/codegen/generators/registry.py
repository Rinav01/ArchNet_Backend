from app.ir.ir_graph import IRGraph

class BaseGenerator:
    def generate(self, ir_graph: IRGraph) -> str:
        """
        Generates the target framework code representation from the ir_graph.
        """
        raise NotImplementedError

class GeneratorRegistry:
    _registry = {}

    @classmethod
    def register(cls, framework: str, generator_cls):
        cls._registry[framework.lower().strip()] = generator_cls

    @classmethod
    def get_generator(cls, framework: str) -> BaseGenerator:
        fw = framework.lower().strip()
        
        # Normalize framework names
        if "pytorch" in fw or "torch" in fw:
            fw = "pytorch"
        elif "tensorflow" in fw or "keras" in fw:
            fw = "tensorflow"
        elif "jax" in fw or "flax" in fw:
            fw = "jax"
        elif "onnx" in fw:
            fw = "onnx"
            
        if fw not in cls._registry:
            # Lazy import to avoid circular dependencies
            if fw == "pytorch":
                from app.codegen.generators.pytorch.generator import PyTorchCompiler
                cls.register("pytorch", PyTorchCompiler)
            elif fw == "tensorflow":
                from app.codegen.generators.tensorflow.compiler import TensorFlowCompiler
                cls.register("tensorflow", TensorFlowCompiler)
            elif fw == "jax":
                from app.codegen.generators.jax.compiler import JAXCompiler
                cls.register("jax", JAXCompiler)
            elif fw == "onnx":
                from app.codegen.generators.onnx.compiler import ONNXCompiler
                cls.register("onnx", ONNXCompiler)
                
        generator_cls = cls._registry.get(fw)
        if not generator_cls:
            raise ValueError(f"No generator registered for framework: {framework}")
        return generator_cls()

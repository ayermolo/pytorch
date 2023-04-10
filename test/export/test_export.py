# Owner(s): ["module: dynamo"]
from torch.testing._internal.common_utils import run_tests, TestCase
from functorch.experimental.control_flow import cond
from torch._export.trace import do_not_use_experimental_export
from torch._export.constraints import add_inline_size_constraint
from torch.fx.experimental.proxy_tensor import make_fx
import torch._dynamo as torchdynamo
from torch._dynamo import config
import torch
import unittest


class TestExport(TestCase):
    @unittest.skip("dynamo failure -> RuntimeError: Could not infer dtype of SymBool")
    def test_export_cond(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return x.cos()

        def foo(x):
            return cond(torch.tensor(x.shape[0] > 4), true_fn, false_fn, [x])

        exported_program = do_not_use_experimental_export(foo, (torch.ones(6, 4, requires_grad=True),))
        print(exported_program.graph_module.graph)

    @unittest.skip("TypeError: <lambda>() missing 1 required positional argument")
    def test_export_simple_model_with_attr(self):
        class Foo(torch.nn.Module):
            def __init__(self, float_val):
                super().__init__()
                self.float_val = float_val

            def forward(self, x):
                y = x + self.float_val
                return y.cos()

        inp = (torch.ones(6, 4, requires_grad=True),)
        mod = Foo(0.5)

        exported_program = do_not_use_experimental_export(mod, inp)
        self.assertEqual(exported_program.fw_module(*inp)[0], mod(*inp))

    @unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
    def test_export_simple_model(self):
        class Foo(torch.nn.Module):
            def __init__(self, float_val):
                super().__init__()
                self.float_val = float_val

            def forward(self, x):
                return x.cos()

        inp = (torch.ones(6, 4, requires_grad=True),)
        mod = Foo(0.5)

        exported_program = do_not_use_experimental_export(mod, inp)
        self.assertEqual(exported_program.fw_module(*inp)[0], mod(*inp))

    @unittest.skip("TypeError: <lambda>() missing 1 required positional argument")
    def test_export_simple_model_buffer_mutation(self):
        class Foo(torch.nn.Module):
            def __init__(self, float_val):
                super().__init__()
                self.register_buffer("buffer1", torch.ones(6, 1))

            def forward(self, x):
                self.buffer1.add_(2)
                return x.cos() + self.buffer1.sin()

        inp = (torch.ones(6, 4, requires_grad=True),)
        mod = Foo(0.5)

        exported_program = do_not_use_experimental_export(mod, inp)
        mutated_buffer, output = exported_program.fw_module(*inp)
        # TODO (tmanlaibaatar) enable this once we figure out
        # how to do buffer mutation
        # self.assertEqual(mutated_buffer.sum().item(), 30)
        self.assertEqual(output, mod(*inp))

    @unittest.skip("RuntimeError: Python 3.11+ not yet supported for torch.compile")
    @unittest.skip("RuntimeError: Windows not yet supported for torch.compile")
    @config.patch(dynamic_shapes=True, capture_dynamic_output_shape_ops=True, specialize_int=True, capture_scalar_outputs=True)
    def test_export_constraints(self):

        def f(x):
            b = x.item()
            add_inline_size_constraint(b, min=2, max=5)
            return torch.full((b, 1), 1)

        inp = (torch.tensor([3]),)
        ref = f(*inp)

        gm, _ = torchdynamo.export(f, *inp, aten_graph=True, tracing_mode="symbolic")
        res = gm(*inp)

        self.assertTrue(torchdynamo.utils.same(ref, res))

        gm = make_fx(f, tracing_mode="symbolic")(*inp)
        res = gm(*inp)
        self.assertTrue(torchdynamo.utils.same(ref, res))

    @unittest.skip("RuntimeError: Python 3.11+ not yet supported for torch.compile")
    @unittest.skip("RuntimeError: Windows not yet supported for torch.compile")
    @config.patch(dynamic_shapes=True, capture_dynamic_output_shape_ops=True, specialize_int=True, capture_scalar_outputs=True)
    def test_export_constraints_error(self):

        def multiple(x):
            b = x.item()
            add_inline_size_constraint(b, min=2, max=3)
            add_inline_size_constraint(b, min=4, max=5)
            return torch.full((b, 1), 1)

        inp = (torch.tensor([3]),)

        with self.assertRaises(torchdynamo.exc.TorchRuntimeError):
            _ = torchdynamo.export(multiple, *inp, aten_graph=True, tracing_mode="symbolic")

        with self.assertRaises(AssertionError):
            _ = make_fx(multiple, tracing_mode="symbolic")(*inp)

if __name__ == '__main__':
    run_tests()

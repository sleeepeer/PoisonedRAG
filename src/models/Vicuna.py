from fastchat.model import load_model, get_conversation_template
import torch

from .Model import Model


class Vicuna(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        self.num_gpus = len(config["params"]["gpus"])
        self.max_gpu_memory = config["params"]["max_gpu_memory"]
        self.revision = config["params"]["revision"]
        self.load_8bit = self.__str_to_bool(config["params"]["load_8bit"])
        self.cpu_offloading = self.__str_to_bool(config["params"]["cpu_offloading"])
        self.debug = self.__str_to_bool(config["params"]["debug"])

        self.repetition_penalty = float(config["params"]["repetition_penalty"])
        if self.load_8bit:
            try:
                import bitsandbytes  # noqa: F401
            except Exception as exc:
                raise RuntimeError(
                    "Vicuna config has load_8bit=True, but bitsandbytes is not installed. "
                    "Install bitsandbytes or set load_8bit=False."
                ) from exc

        try:
            self.model, self.tokenizer = load_model(
                model_path=self.name,
                device=self.device,
                num_gpus=self.num_gpus,
                max_gpu_memory=self.max_gpu_memory,
                # dtype=torch.float16,
                load_8bit=self.load_8bit,
                cpu_offloading=self.cpu_offloading,
                revision=self.revision,
                debug=self.debug,
            )
        except Exception as exc:
            message = str(exc).lower()
            if "out of memory" in message:
                raise RuntimeError(
                    "Vicuna model load hit GPU OOM. This GPU has limited VRAM for 7B full precision. "
                    "Use a quantized setup (8-bit/4-bit), a smaller local model, or run on a larger GPU."
                ) from exc
            raise
        self.input_device = self._resolve_input_device()
        print(
            "[Vicuna] Runtime settings:",
            f"requested_device={self.device}",
            f"runtime_input_device={self.input_device}",
            f"cuda_available={torch.cuda.is_available()}",
            f"load_8bit={self.load_8bit}",
            f"cpu_offloading={self.cpu_offloading}",
        )
        if hasattr(self.model, "hf_device_map"):
            device_map = getattr(self.model, "hf_device_map", {})
            unique_devices = sorted({str(v) for v in device_map.values()})
            print(f"[Vicuna] hf_device_map devices: {unique_devices}")

    def __str_to_bool(self, s):
        if type(s) == str:
            if s.lower() == 'true':
                return True
            elif s.lower() == 'false':
                return False
        raise ValueError(f'{s} is not a valid boolean')

    def _to_torch_device(self, device_value):
        if isinstance(device_value, torch.device):
            return device_value
        if isinstance(device_value, int):
            return torch.device(f"cuda:{device_value}")
        if isinstance(device_value, str):
            if device_value.startswith("cuda") or device_value == "cpu":
                return torch.device(device_value)
        return None

    def _resolve_input_device(self):
        if hasattr(self.model, "device"):
            resolved = self._to_torch_device(self.model.device)
            if resolved is not None:
                return resolved

        if hasattr(self.model, "hf_device_map"):
            device_map = getattr(self.model, "hf_device_map", {})
            for device_value in device_map.values():
                resolved = self._to_torch_device(device_value)
                if resolved is not None and str(resolved) != "cpu":
                    return resolved

        try:
            return next(self.model.parameters()).device
        except StopIteration:
            pass
        except Exception:
            pass

        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def query(self, msg):
        conv = get_conversation_template(self.name)
        conv.append_message(conv.roles[0], msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = self.tokenizer([prompt]).input_ids
        input_tensor = torch.as_tensor(input_ids, device=self.input_device)
        output_ids = self.model.generate(
            input_tensor,
            do_sample=True,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_output_tokens,
        )

        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]) :]
        response = self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        return response

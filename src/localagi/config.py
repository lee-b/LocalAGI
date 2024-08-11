from dataclasses import dataclass


@dataclass
class Config:
    # globals (for now) (which we initialize later)
    SYSTEM_PROMPT = None
    DEFAULT_API_BASE = "http://api:8080"

    DEFAULT_PROMPT = None
    LOCALAI_API_BASE = None
    TTS_API_BASE = None
    IMAGE_API_BASE = None
    EMBEDDINGS_API_BASE = None
    STABLEDIFFUSION_MODEL = None
    STABLEDIFFUSION_PROMPT = None
    FUNCTIONS_MODEL = "functions"
    EMBEDDINGS_MODEL = None
    LLM_MODEL = "gpt-4"
    VOICE_MODEL = "en-us-kathleen-low.onnx"
    STABLEDIFFUSION_MODEL = "stablediffusion"
    STABLEDIFFUSION_PROMPT = None
    PERSISTENT_DIR = None

    @classmethod
    def from_environ_and_args(cls, environ, args) -> 'Config':
        """populates some global settings from the runtime context
        
        (until we can replace these with locals)
        """

        cfg = cls()

        cfg.DEFAULT_API_BASE = environ.get("DEFAULT_API_BASE", cfg.DEFAULT_API_BASE)
        cfg.DEFAULT_PROMPT="floating hair, portrait, ((loli)), ((one girl)), cute face, hidden hands, asymmetrical bangs, beautiful detailed eyes, eye shadow, hair ornament, ribbons, bowties, buttons, pleated skirt, (((masterpiece))), ((best quality)), colorful|((part of the head)), ((((mutated hands and fingers)))), deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, Octane renderer, lowres, bad anatomy, bad hands, text"
        cfg.EMBEDDINGS_API_BASE = args.embeddings_api_base
        cfg.EMBEDDINGS_MODEL = environ.get("EMBEDDINGS_MODEL", args.embeddings_model)
        cfg.FUNCTIONS_MODEL = environ.get("FUNCTIONS_MODEL", args.functions_model)
        cfg.IMAGE_API_BASE = args.images_api_base
        cfg.LLM_MODEL = environ.get("LLM_MODEL", args.llm_model)
        cfg.LOCALAI_API_BASE = args.localai_api_base
        cfg.PERSISTENT_DIR = environ.get("PERSISTENT_DIR", "/data")
        cfg.STABLEDIFFUSION_MODEL = environ.get("STABLEDIFFUSION_MODEL", args.stablediffusion_model)
        cfg.STABLEDIFFUSION_PROMPT = environ.get("STABLEDIFFUSION_PROMPT", args.stablediffusion_prompt)

        if environ.get("SYSTEM_PROMPT") or args.system_prompt:
            cfg.SYSTEM_PROMPT = environ.get("SYSTEM_PROMPT", args.system_prompt)
        else:
            cfg.SYSTEM_PROMPT = ""

        cfg.TTS_API_BASE = args.tts_api_base
        cfg.VOICE_MODEL= environ.get("TTS_MODEL", args.tts_model)

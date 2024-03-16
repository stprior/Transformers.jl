#for use from repl
using PyCall
using HuggingFaceApi
using Transformers.HuggingFace
using TextEncodeBase

model_name = "EleutherAI/pythia-14m"
hgf_trf = pyimport("transformers")
pyconfig = hgf_trf.AutoConfig.from_pretrained(model_name, layer_norm_eps = 1e-9, layer_norm_epsilon = 1e-9)
py_hgf_tkr = hgf_trf.AutoTokenizer.from_pretrained(model_name, config = pyconfig)

jlconfig = HuggingFace.load_config(model_name)
jl_hgf_tkr = HuggingFace.load_tokenizer(model_name; config = jlconfig)
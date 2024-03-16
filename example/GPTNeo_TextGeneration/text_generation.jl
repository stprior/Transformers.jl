using Flux
using Transformers
using Transformers.TextEncoders
using Transformers.HuggingFace

 

function temp_softmax(logits; temperature=1.2)
    return softmax(logits ./ temperature)
end

function top_k_sample(probs; k=10)
    indexes = partialsortperm(probs, 1:k, rev=true)
    index = rand(collect(indexes))
    return index
end

function generate_text(context=""; max_length=50)
    textenc = hgf"EleutherAI/pythia-14m:tokenizer"
    model = hgf"EleutherAI/pythia-14m:forcausallm"

    tokens = encode(textenc, context).token
    ids = tokens.onehots
    ends_id = lookup(textenc.vocab, textenc.endsym)
    for i in 1:max_length
        input = (; token = tokens)
        outputs = model(input)
        logits = @view outputs.logit[:, end, 1]
        probs = temp_softmax(logits)
        new_id = top_k_sample(probs)[1]
        push!(ids, new_id)
        new_id == ends_id && break
    end
    return decode(textenc, tokens)
end

function generate(prompt, max_length)
    text_token = generate_text(prompt; max_length=max_length)
    gen_text = join(text_token)
    print("\n\nGenerated Text: ")
    println(gen_text)
end

generate("The capital of Ireland is", 20)

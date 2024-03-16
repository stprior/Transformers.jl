using TextEncodeBase: getvalue, nestedcall

function test_tokenizer(name, corpus; output = nothing)
    global torch, hgf_trf, config, vocab_size, config, pyconfig
    @info "Validate $name tokenizer with corpus $corpus"
    @testset "Tokenizer" begin
        isfile(corpus) || error("corpus file $corpus do not exist.")

        @info "Loading tokenizer in Python"
        hgf_tkr = @tryrun begin
            hgf_trf.AutoTokenizer.from_pretrained(name, config = pyconfig)
        end "Failed to load the tokenizer in Python"
        @info "Python tokenizer loaded successfully"

        @info "Loading tokenizer in Julia"
        tkr = @tryrun begin
            HuggingFace.load_tokenizer(name; config)
        end "Failed to load the tokenizer in Julia"
        @info "Julia tokenizer loaded successfully"

        @info "Comparing Tokenizers"
        #hgf_tkr.vocab is a Python Dict of string, int
        mismatches=0
        i=0
        for (py_token, py_token_id) in hgf_tkr.vocab
            try
                i=i+1
                @info "Testing token $i: $py_token"
                jl_token_id = lookup(tkr.vocab, py_token)                    
                if jl_token_id != (py_token_id+1)
                    @warn "Token \"$py_token\" has different id in Julia ($jl_token_id) and Python ($py_token_id)"
                    mismatches += 1
                end
            catch e
                @warn "Token \"$py_token\" caused error $e"
                mismatches += 1                    
            end
            
        end
        @test mismatches <= 5

        @info "Testing: Tokenizer"
        fd = nothing
        out_fd = nothing
        prev_line = nothing
        try
            fd = open(corpus)
            isnothing(output) || (out_fd = open(output, "w+"))
            for line in eachline(fd)
                # single input
                isempty(line) && continue
                jl_tokens = TextEncodeBase.getvalue.(TextEncodeBase.tokenize(tkr, line))
                py_tokens = hgf_tkr.tokenize(line)
                @test jl_tokens == py_tokens
                jl_indices = collect(reinterpret(Int32, encode(tkr, line).token))
                py_indices = collect(hgf_tkr(line)["input_ids"]) .+ 1
                jl_ind_len = length(jl_indices)
                py_ind_len = length(py_indices)
                if jl_ind_len > py_ind_len
                    @test jl_indices[begin:py_ind_len] == py_indices
                elseif py_ind_len > jl_ind_len
                    @test jl_indices == py_indices[begin:jl_ind_len]
                else
                    @test jl_indices == py_indices
                end

                single_pass = jl_tokens == py_tokens
                if !single_pass
                    println("Failed: ", repr(line))
                    isnothing(out_fd) || println(out_fd, line)
                end
                # pair input
                if !isnothing(prev_line) && single_pass
                    pair_jl_tokens =
                        vcat(nestedcall(getvalue, TextEncodeBase.tokenize(tkr, [[prev_line, line]]))[]...)
                    pair_py_tokens = hgf_tkr.tokenize(prev_line, line)
                    @test pair_jl_tokens == pair_py_tokens
                    pair_jl_indices = reshape(
                        collect(reinterpret(Int32, encode(tkr, [[prev_line, line]]).token)), :)
                    pair_py_indices = collect(hgf_tkr(prev_line, line)["input_ids"]) .+ 1
                    pair_jl_ind_len = length(pair_jl_indices)
                    pair_py_ind_len = length(pair_py_indices)
                    if pair_jl_ind_len > pair_py_ind_len
                        @test pair_jl_indices[begin:pair_py_ind_len] == pair_py_indices
                    elseif pair_py_ind_len > pair_jl_ind_len
                        @test pair_jl_indices == pair_py_indices[begin:pair_jl_ind_len]
                    else
                        @test pair_jl_indices == pair_py_indices
                    end
                end
                single_pass && (prev_line = line)
            end
        catch e
            isnothing(fd) || close(fd)
            isnothing(out_fd) || close(out_fd)
            rethrow(e)
        finally
            isnothing(fd) || close(fd)
            isnothing(out_fd) || close(out_fd)
        end
    end
end

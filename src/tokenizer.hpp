// SentencePiece-style BPE for the TinyLlama (Llama 2) 32k vocabulary.
#pragma once
#include <string>
#include <unordered_map>
#include <vector>

namespace tl {

class Tokenizer {
public:
    explicit Tokenizer(const std::string& path);

    // Encodes plain text. add_prefix prepends the SentencePiece "▁" (space marker).
    std::vector<int> encode(const std::string& text, bool add_prefix) const;

    // Zephyr chat template used by TinyLlama-Chat and by Ollama's tinyllama:
    //   <s><|system|>\n{system}</s>\n<|user|>\n{user}</s>\n<|assistant|>\n
    std::vector<int> encode_first_turn(const std::string& system, const std::string& user) const;
    //   </s>\n<|user|>\n{user}</s>\n<|assistant|>\n   (follow-up turns in chat mode)
    std::vector<int> encode_next_turn(const std::string& user) const;

    std::string decode(int token) const;

private:
    std::vector<std::string> vocab_;
    std::vector<float> scores_;
    std::unordered_map<std::string, int> lookup_;
};

}  // namespace tl

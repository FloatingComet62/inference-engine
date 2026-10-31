#include "tokenizer.hpp"
#include "config.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace tl {

namespace {
const std::string SPACE_MARK = "\xE2\x96\x81";   // "▁"
constexpr int BYTE_TOKEN_OFFSET = 3;              // <0x00> is token 3

size_t utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c >> 5) == 0x6) return 2;
    if ((c >> 4) == 0xE) return 3;
    if ((c >> 3) == 0x1E) return 4;
    return 1;
}
}  // namespace

Tokenizer::Tokenizer(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open tokenizer: " + path + " (run `make setup`)");
    char magic[4];
    uint32_t n = 0;
    if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "TLT1", 4) != 0 ||
        std::fread(&n, 4, 1, f) != 1 || n != (uint32_t)VOCAB) {
        std::fclose(f);
        throw std::runtime_error("bad tokenizer file: " + path);
    }
    vocab_.resize(n);
    scores_.resize(n);
    for (uint32_t i = 0; i < n; i++) {
        uint32_t len = 0;
        if (std::fread(&scores_[i], 4, 1, f) != 1 || std::fread(&len, 4, 1, f) != 1) break;
        vocab_[i].resize(len);
        if (len && std::fread(&vocab_[i][0], 1, len, f) != len) break;
        lookup_.emplace(vocab_[i], (int)i);
    }
    std::fclose(f);
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_prefix) const {
    std::string s = add_prefix ? SPACE_MARK : "";
    for (char c : text) {
        if (c == ' ') s += SPACE_MARK; else s += c;
    }

    // 1. one token per UTF-8 character, falling back to raw byte tokens
    std::vector<int> ids;
    for (size_t i = 0; i < s.size();) {
        size_t len = utf8_len((unsigned char)s[i]);
        if (i + len > s.size()) len = 1;
        auto it = lookup_.find(s.substr(i, len));
        if (it != lookup_.end()) ids.push_back(it->second);
        else for (size_t j = 0; j < len; j++) ids.push_back((unsigned char)s[i + j] + BYTE_TOKEN_OFFSET);
        i += len;
    }

    // 2. repeatedly merge the adjacent pair with the best (highest) merge score
    std::string buf;
    while (true) {
        float best = -1e30f;
        int best_id = -1, best_i = -1;
        for (size_t i = 0; i + 1 < ids.size(); i++) {
            buf = vocab_[ids[i]];
            buf += vocab_[ids[i + 1]];
            auto it = lookup_.find(buf);
            if (it != lookup_.end() && scores_[it->second] > best) {
                best = scores_[it->second];
                best_id = it->second;
                best_i = (int)i;
            }
        }
        if (best_i < 0) break;
        ids[best_i] = best_id;
        ids.erase(ids.begin() + best_i + 1);
    }
    return ids;
}

std::vector<int> Tokenizer::encode_first_turn(const std::string& system, const std::string& user) const {
    std::vector<int> ids{BOS};
    auto add = [&](const std::vector<int>& v) { ids.insert(ids.end(), v.begin(), v.end()); };
    add(encode("<|system|>\n" + system, true));
    ids.push_back(EOS);
    add(encode("\n<|user|>\n" + user, false));
    ids.push_back(EOS);
    add(encode("\n<|assistant|>\n", false));
    return ids;
}

std::vector<int> Tokenizer::encode_next_turn(const std::string& user) const {
    std::vector<int> ids{EOS};
    auto add = [&](const std::vector<int>& v) { ids.insert(ids.end(), v.begin(), v.end()); };
    add(encode("\n<|user|>\n" + user, false));
    ids.push_back(EOS);
    add(encode("\n<|assistant|>\n", false));
    return ids;
}

std::string Tokenizer::decode(int token) const {
    const std::string& piece = vocab_[token];
    unsigned int byte = 0;
    if (piece.size() == 6 && std::sscanf(piece.c_str(), "<0x%02X>", &byte) == 1)
        return std::string(1, (char)byte);
    std::string out;
    for (size_t i = 0; i < piece.size();) {
        if (piece.compare(i, SPACE_MARK.size(), SPACE_MARK) == 0) { out += ' '; i += SPACE_MARK.size(); }
        else out += piece[i++];
    }
    return out;
}

}  // namespace tl

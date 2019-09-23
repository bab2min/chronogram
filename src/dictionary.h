#pragma once

#include <string>
#include <map>
#include <vector>
#include <mutex>

template<class KeyType = std::string, class ValueType = int32_t>
class WordDictionary
{
protected:
	std::map<KeyType, ValueType> word2id;
	std::vector<KeyType> id2word;
public:

	WordDictionary() {}
	WordDictionary(const WordDictionary& o) : word2id(o.word2id), id2word(o.id2word) {}
	WordDictionary(WordDictionary&& o)
	{
		std::swap(word2id, o.word2id);
		std::swap(id2word, o.id2word);
	}

	WordDictionary& operator=(WordDictionary&& o)
	{
		std::swap(word2id, o.word2id);
		std::swap(id2word, o.id2word);
		return *this;
	}

	enum { npos = (ValueType)-1 };
	ValueType add(const KeyType& str)
	{
		if (word2id.emplace(str, word2id.size()).second) id2word.emplace_back(str);
		return word2id.size() - 1;
	}

	ValueType getOrAdd(const KeyType& str)
	{
		auto it = word2id.find(str);
		if (it != word2id.end()) return it->second;
		return add(str);
	}

	ValueType get(const KeyType& str) const
	{
		auto it = word2id.find(str);
		if (it != word2id.end()) return it->second;
		return npos;
	}

	const KeyType& getStr(int id) const
	{
		return id2word[id];
	}

	size_t size() const { return id2word.size(); }

	void writeToFile(std::ostream& str) const
	{
		uint32_t vocab = id2word.size();
		str.write((const char*)&vocab, 4);
		for (auto w : id2word)
		{
			uint32_t len = w.size();
			str.write((const char*)&len, 4);
			str.write(&w[0], len);
		}
	}

	template<class _Istream>
	void readFromFile(_Istream& str)
	{
		uint32_t vocab;
		str.read((char*)&vocab, 4);
		id2word.resize(vocab);
		for (auto& w : id2word)
		{
			uint32_t len;
			str.read((char*)&len, 4);
			w.resize(len);
			str.read(&w[0], len);
		}

		for (size_t i = 0; i < id2word.size(); ++i)
		{
			word2id[id2word[i]] = i;
		}
	}

	const std::vector<KeyType>& getKeys() const { return id2word; }
};

#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <iterator>

void Tokenize(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters = " ") {
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  std::string::size_type pos     = str.find_first_of(delimiters, lastPos);

  while (std::string::npos != pos || std::string::npos != lastPos) {
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    lastPos = str.find_first_not_of(delimiters, pos);
    pos = str.find_first_of(delimiters, lastPos);
  }
}

class Model {
  friend class Decoder;
public:
  Model(const std::string& file);
  void print(std::ostream *os);

private:
  static const double kDefaultWeight;
  enum { Weight = 0, Attr = 1, Label = 2};
  std::string fileName;
  std::map<std::string, int> labels;
  std::map<std::string, std::vector<double> > weights;
};

class Decoder {
public:
  Decoder(const std::string& modelFile, const std::string& targetFile, const std::string& originalFile);
  void print_model(std::ostream *os);
  void decode(std::ostream *os);

private:
  Model model;
  std::string targetFileName;
  std::string originalFileName;
  std::vector<double> calcScores(std::vector<std::string> attrs, const std::string& prev);
  std::pair<std::string, double> predictAndCalcMergin(std::vector<double> scores);
};

Model::Model(const std::string& file)
  :fileName(file), labels(), weights() {
    std::ifstream modelFile(fileName.c_str()); 
    std::string line;
    if (modelFile.is_open()) {
      int i = 0;
      while (modelFile.good()) {
        getline(modelFile, line);
        std::vector<std::string> tokens;
        Tokenize(line, tokens, "\t");
        if (tokens.size() != 0) {
          if (tokens[0] == "@classias") {
          } else if (tokens[0] == "@label") {
            labels[tokens[1]] = i++;
          } else {
            if (weights[tokens[Attr]].empty()) {
              for (int j = 0; j < labels.size(); j++) {
                weights[tokens[Attr]].push_back(kDefaultWeight);
              }
            }
            weights[tokens[Attr]][labels[tokens[Label]]] = strtod(tokens[Weight].c_str(), NULL);
          }
        }
      }
    }
    modelFile.close();
}

void Model::print(std::ostream *os) {
  std::map<std::string, int>::iterator lit;
  std::map<std::string, std::vector<double> >::iterator wit;
  for (lit = labels.begin(); lit != labels.end(); lit++) {
    *os << (*lit).first << ": " << (*lit).second << std::endl;
  }
  for (wit = weights.begin(); wit != weights.end(); wit++) {
    *os << (*wit).first << " => ";
    for (int i = 0; i < (*wit).second.size(); i++) {
      *os << (*wit).second[i] << " ";
    }
    *os << std::endl;
  }
}

const double Model::kDefaultWeight = 0.;

Decoder::Decoder(const std::string& modelFile, const std::string& targetFile, const std::string& originalFile)
  :model(modelFile), targetFileName(targetFile), originalFileName(originalFile) {}

void Decoder::decode(std::ostream *os) {
  std::ifstream targetFile(targetFileName.c_str());
  std::ifstream originalFile(originalFileName.c_str());
  std::string line, oline;
  std::string prev("");

  if (targetFile.is_open() && originalFile.is_open()) {
    getline(originalFile, oline); // avoid first line
    while (targetFile.good() && originalFile.good()) {
      getline(targetFile, line);
      getline(originalFile, oline);
      std::vector<std::string> tokens, otokens;
      Tokenize(line, tokens, "\t");
      Tokenize(oline, otokens, "\t");

      if (!tokens.empty()) { 
        std::string ans = tokens[0];
        tokens.erase(tokens.begin());

        std::vector<double> scores = calcScores(tokens, prev);
        std::pair<std::string, double> pred_mergin = predictAndCalcMergin(scores);

        *os << ans << "\t" << pred_mergin.first << "\t" << pred_mergin.second << "\t" << otokens[1] << std::endl;
        prev = std::string(pred_mergin.first);
      } else {
        *os << std::endl;
        prev = "";
      }
    }
  }
  targetFile.close();
}

void Decoder::print_model(std::ostream *os) {
  *os << "target: " << targetFileName << std::endl;
  model.print(os);
}

std::vector<double> Decoder::calcScores(std::vector<std::string> attrs, const std::string& prev) {
  std::vector<double> scores (model.labels.size(), 0.);
  std::vector<std::string>::iterator it;
  std::map<std::string, std::vector<double> >::iterator wit;
  std::map<std::string, int>::iterator lit;

  if ((wit = model.weights.find("$y[-1]=" + prev)) != model.weights.end()) {
    for (lit = model.labels.begin(); lit != model.labels.end(); lit++) {
      scores[(*lit).second] += (*wit).second[(*lit).second];
    }
  }
  
  for (it = attrs.begin(); it < attrs.end(); it++) {
    if ((wit = model.weights.find(*it)) != model.weights.end()) {
      std::vector<double> table = (*wit).second;
      for (lit = model.labels.begin(); lit != model.labels.end(); lit++) {
        scores[(*lit).second] += table[(*lit).second];
      }
    }
  }
  return scores;
}

std::pair<std::string, double> Decoder::predictAndCalcMergin(std::vector<double> scores) {
  std::vector<double>::iterator it;
  std::pair<double, int> max (0, 0);
  double next = 0;

  for (it = scores.begin(); it < scores.end(); it++) {
    if (*it > next) {
      if (*it > max.first) {
        next = max.first;
        max.first = *it;
        max.second = it - scores.begin();
      } else {
        next = *it;
      }
    }
  }

  std::map<std::string, int>::iterator lit;
  std::string predict;

  for (lit = model.labels.begin(); lit != model.labels.end(); lit++) {
    if ((*lit).second == max.second) predict = (*lit).first;
  }
  return std::pair<std::string, double> (predict, max.first - next);
}

void run_test() {
  Decoder mydecoder("tests/train.base.1.svm.model2", "tests/test.base", "tests/test.txt");
  // mydecoder.print_model(&std::cout);
  mydecoder.decode(&std::cout);
  // Model mymodel2("tests/train.base.1.svm.model2");
  // mymodel2.print(&std::cout);
}

int main(int argc, char *argv[]) {
  std::cout.precision(8);
  if (argc == 1) {
    run_test();
  }
}

#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <iterator>


void Tokenize(const std::string& str,
              std::vector<std::string>& tokens,
              const std::string& delimiters = " ") {
  // Skip delimiters at beginning.
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  // Find first "non-delimiter".
  std::string::size_type pos     = str.find_first_of(delimiters, lastPos);

  while (std::string::npos != pos || std::string::npos != lastPos)
  {
    // Found a token, add it to the vector.
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    // Skip delimiters.  Note the "not_of"
    lastPos = str.find_first_not_of(delimiters, pos);
    // Find next "non-delimiter"
    pos = str.find_first_of(delimiters, lastPos);
  }
}

class Model {
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
  Decoder(const std::string& modelFile, const std::string& targetFile);
  void print_model(std::ostream *os);

private:
  Model model;
  std::string targetFileName;
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

Decoder::Decoder(const std::string& modelFile, const std::string& targetFile)
  :model(modelFile), targetFileName(targetFile) {}

void Decoder::print_model(std::ostream *os) {
  *os << "target: " << targetFileName << std::endl;
  model.print(os);
}

void run_test() {
  Decoder mydecoder("tests/train.sample", "tests/test.base");
  mydecoder.print_model(&std::cout);
  // Model mymodel2("tests/train.base.1.svm.model2");
  // mymodel2.print(&std::cout);
}

int main(int argc, char *argv[]) {
  if (argc == 1) {
    run_test();
  }
}

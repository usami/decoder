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
  std::string fileName;
  std::vector<std::string> labels;
  std::map<std::string, std::vector<double> > weights;
};

Model::Model(const std::string& file)
  :fileName(file), labels(), weights() {
    std::ifstream modelFile(fileName.c_str()); 
    std::string line;
    if (modelFile.is_open()) {
      while (modelFile.good()) {
        getline(modelFile, line);
        std::vector<std::string> tokens;
        Tokenize(line, tokens, "\t");
        if (tokens.size() != 0) {
          if (tokens[0] == "@classias") {
          } else if (tokens[0] == "@label") {
            labels.push_back(tokens[1]);
          } else {
            weights[tokens[1]].push_back(strtod(tokens[0].c_str(), NULL));
          }
        }
      }
    }
    modelFile.close();
}

void Model::print(std::ostream *os) {
  std::map<std::string, std::vector<double> >::iterator it;
  for (int i = 0; i < labels.size(); i++) {
    *os << labels[i] << std::endl;
  }
  for (it = weights.begin(); it != weights.end(); it++) {
    *os << (*it).first << " => ";
    for (int i = 0; i < (*it).second.size(); i++) {
      *os << (*it).second[i] << " ";
    }
    *os << std::endl;
  }
}

void run_test() {
  Model mymodel("tests/train.sample");
  mymodel.print(&std::cout);
}

int main(int argc, char *argv[]) {
  if (argc == 1) {
    run_test();
  }
}

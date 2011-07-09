#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <iterator>

using namespace std;

void Tokenize(const string& str, vector<string>& tokens, const string& delimiters = " ") {
  string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  string::size_type pos     = str.find_first_of(delimiters, lastPos);

  while (string::npos != pos || string::npos != lastPos) {
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    lastPos = str.find_first_not_of(delimiters, pos);
    pos = str.find_first_of(delimiters, lastPos);
  }
}

class Model {
  friend class Decoder;
public:
  Model(const string& file);
  void print(ostream *os);

private:
  static const double kDefaultWeight;
  enum { Weight = 0, Attr = 1, Label = 2};
  string fileName;
  map<string, int> labels;
  map<string, vector<double> > weights;
};

class Decoder {
public:
  Decoder(const string& modelFile, const string& targetFile, const string& originalFile);
  void print_model(ostream *os);
  void decode(ostream *os);

private:
  Model model;
  string targetFileName;
  string originalFileName;
  vector<double> calcScores(vector<string> attrs, const string& prev);
  pair<string, double> predictAndCalcMergin(vector<double> scores);
  void printLog(vector<double> &scores, vector<string> &attrs, ostream *os, string &predict, string &ans);
};

Model::Model(const string& file)
  :fileName(file), labels(), weights() {
    ifstream modelFile(fileName.c_str()); 
    string line;
    if (modelFile.is_open()) {
      int i = 0;
      while (modelFile.good()) {
        getline(modelFile, line);
        vector<string> tokens;
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

void Model::print(ostream *os) {
  map<string, int>::iterator lit;
  map<string, vector<double> >::iterator wit;
  for (lit = labels.begin(); lit != labels.end(); lit++) {
    *os << (*lit).first << ": " << (*lit).second << endl;
  }
  for (wit = weights.begin(); wit != weights.end(); wit++) {
    *os << (*wit).first << " => ";
    for (int i = 0; i < (*wit).second.size(); i++) {
      *os << (*wit).second[i] << " ";
    }
    *os << endl;
  }
}

const double Model::kDefaultWeight = 0.;

Decoder::Decoder(const string& modelFile, const string& targetFile, const string& originalFile)
  :model(modelFile), targetFileName(targetFile), originalFileName(originalFile) {}

void Decoder::decode(ostream *os) {
  ifstream targetFile(targetFileName.c_str());
  ifstream originalFile(originalFileName.c_str());
  string line, oline;
  string prev("");

  if (targetFile.is_open() && originalFile.is_open()) {
    getline(originalFile, oline); // avoid first line
    while (targetFile.good() && originalFile.good()) {
      getline(targetFile, line);
      getline(originalFile, oline);
      vector<string> tokens, otokens;
      Tokenize(line, tokens, "\t");
      Tokenize(oline, otokens, "\t");

      if (!tokens.empty()) { 
        string ans = tokens[0];
        tokens.erase(tokens.begin());
        tokens.push_back("__BIAS__");

        vector<double> scores = calcScores(tokens, prev);
        pair<string, double> pred_mergin = predictAndCalcMergin(scores);

        *os << ans << "\t" << pred_mergin.first << "\t" << pred_mergin.second << "\t" << otokens[1] << endl;
        if (ans != pred_mergin.first) {
          printLog(scores, tokens, &cerr, pred_mergin.first, ans);
        }
        prev = string(pred_mergin.first);
      } else {
        *os << endl;
        prev = "";
      }
    }
  }
  targetFile.close();
}

void Decoder::printLog(vector<double> &scores, vector<string> &attrs, ostream *os, string &predict, string &ans) {
  for (int i = -2; i < 3; i++) {
    string prefix = "w[";
    char w[4];
    sprintf(w, "%d]=", i);
    prefix += w;
    for (vector<string>::iterator it = attrs.begin(); it != attrs.end(); it++) {
      if (!(*it).compare(0, prefix.size(), prefix)) {
        if (i == 0) *os << "[";
        *os << (*it).substr(prefix.size());
        if (i == 0) *os << "]";
      }
    }
    *os << " ";
  }
  *os << endl;
  *os << "Prediction: " << predict << "\t" << "Answer: " << ans << endl;
  for (map<string, int>::iterator it = model.labels.begin(); it != model.labels.end(); it++) {
    *os << (*it).first << ": " << scores[(*it).second] << "\t";
  }
  *os << endl;

  for (map<string, int>::iterator it = model.labels.begin(); it != model.labels.end(); it++) {
    *os << "--------------------------" << endl;
    map<string, vector<double> >::iterator wit;
    map<double, string> value_attrs;
    for (vector<string>::iterator ait = attrs.begin(); ait != attrs.end(); ait++) {
      if ((wit = model.weights.find(*ait)) != model.weights.end()) {
        value_attrs[(*wit).second[(*it).second]] = *ait;
        // *os << (*ait) << ": " << (*wit).second[(*it).second] << endl;
      }
    }
    int i = 1;
    *os << (*it).first << " Label Feature Top" << endl;
    for (map<double, string>::reverse_iterator mit = value_attrs.rbegin(); mit != value_attrs.rend() && i < 11; mit++) {
      *os << "[" << i << "] " << (*mit).second << ": " <<  (*mit).first << endl;
      i++;
    }
    i = 1;
    *os << endl;
    *os << (*it).first << " Label Feature Worst" << endl;
    for (map<double, string>::iterator mit = value_attrs.begin(); mit != value_attrs.end() && i < 11; mit++) {
      *os << "[" << i << "] " << (*mit).second << ": " <<  (*mit).first << endl;
      i++;
    }
  }
  *os << endl;
}

void Decoder::print_model(ostream *os) {
  *os << "target: " << targetFileName << endl;
  model.print(os);
}

vector<double> Decoder::calcScores(vector<string> attrs, const string& prev) {
  vector<double> scores (model.labels.size(), 0.);
  vector<string>::iterator it;
  map<string, vector<double> >::iterator wit;
  map<string, int>::iterator lit;

  if ((wit = model.weights.find("$y[-1]=" + prev)) != model.weights.end()) {
    for (lit = model.labels.begin(); lit != model.labels.end(); lit++) {
      scores[(*lit).second] += (*wit).second[(*lit).second];
      // cout << (*lit).first << "\t" << (*wit).second[(*lit).second] << "\t" << scores[(*lit).second] << endl;
    }
  }
  
  for (it = attrs.begin(); it < attrs.end(); it++) {
    if ((wit = model.weights.find(*it)) != model.weights.end()) {
      vector<double> table = (*wit).second;
      for (lit = model.labels.begin(); lit != model.labels.end(); lit++) {
        scores[(*lit).second] += table[(*lit).second];
        // cout << *it << "\t" << table[(*lit).second] << "\t" << scores[(*lit).second] << endl;
      }
    }
  }
  // for (int i = 0; i < scores.size(); i++) {
    // cout << scores[i] << endl;
  // }
  return scores;
}

pair<string, double> Decoder::predictAndCalcMergin(vector<double> scores) {
  vector<double>::iterator it;
  pair<double, int> max (-100, -1);
  double next = - 100;

  for (it = scores.begin(); it < scores.end(); it++) {
    if (*it > next) {
      // cout << *it << " " << next << endl;
      if (*it > max.first) {
        next = max.first;
        max.first = *it;
        max.second = it - scores.begin();
      } else {
        next = *it;
      }
    }
  }

  map<string, int>::iterator lit;
  string predict;

  for (lit = model.labels.begin(); lit != model.labels.end(); lit++) {
    if ((*lit).second == max.second) predict = (*lit).first;
  }
  // cout << "next: " << next << endl;
  return pair<string, double> (predict, max.first - next);
}

void run_test() {
  //Decoder mydecoder("tests/train.hw.1.1.svm.model2", "tests/test.base", "tests/dev.bionlp11epi.txt");
  Decoder mydecoder("tests/c1model2", "tests/test.base", "tests/dev.bionlp11epi.txt");
  // mydecoder.print_model(&cout);
  mydecoder.decode(&cout);
  // Model mymodel2("tests/train.base.1.svm.model2");
  // mymodel2.print(&cout);
}

int main(int argc, char *argv[]) {
  cout.precision(7);
  if (argc == 1) {
    run_test();
  } else if (argc == 4) {
    Decoder decoder(argv[1], argv[2], argv[3]);
    decoder.decode(&cout);
  } else {
    cerr << "Usage: " << argv[0] << " [model file name] [test feature file name] [test file]" << endl;
  }
}

//
//  CRF++ -- Yet Another CRF toolkit
//
//  $Id: encoder.cpp 1601 2007-03-31 09:47:18Z taku $;
//
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//
#include <fstream>
#include "param.h"
#include "encoder.h"
#include "timer.h"
#include "tagger.h"
#include "lbfgs.h"
#include "common.h"
#include "feature_index.h"
#include "scoped_ptr.h"
#include "thread.h"
#include "mc_sample.h"
#include "dasmc/dasmc_train.h"

namespace {
bool toLower(std::string *s) {
  for (size_t i = 0; i < s->size(); ++i) {
    char c = (*s)[i];
    if ((c >= 'A') && (c <= 'Z')) {
      c += 'a' - 'A';
      (*s)[i] = c;
    }
  }
  return true;
}
}

namespace CRFPP {

class CRFEncoderThread: public thread {
 public:
  // training sequence data
  TaggerImpl **x;
  // start index
  unsigned short start_i;
  // #thread
  unsigned short thread_num;
  // sequence error
  int zeroone;
  // node error
  int err;
  // number of sequences
  size_t size;
  // objective: log-likelihood
  double obj;
  // expected value of feature: E[f_k]
  std::vector<double> expected;

  void run() {
    obj = 0.0;
    err = zeroone = 0;
    std::fill(expected.begin(), expected.end(), 0.0);
    for (size_t i = start_i; i < size; i += thread_num) {
      // RULE: gradient of log partition function equals to expectation of features
      // gradient returns the log-likelihood of current sequence
      obj += x[i]->gradient(&expected[0]);
      int error_num = x[i]->eval();
      err += error_num;
      if (error_num) ++zeroone;
    }
  }
};

bool runMIRA(const std::vector<TaggerImpl* > &x,
             EncoderFeatureIndex *feature_index,
             double *alpha,
             size_t maxitr,
             float C,
             double eta,
             unsigned short shrinking_size,
             unsigned short thread_num) {
  std::vector<unsigned char> shrink(x.size());
  std::vector<float> upper_bound(x.size());
  std::vector<double> expected(feature_index->size());

  std::fill(upper_bound.begin(), upper_bound.end(), 0.0);
  std::fill(shrink.begin(), shrink.end(), 0);

  int converge = 0;
  int all = 0;
  for (size_t i = 0; i < x.size(); ++i)  all += x[i]->size();

  for (size_t itr = 0; itr < maxitr; ++itr) {
    int zeroone = 0;
    int err = 0;
    int active_set = 0;
    int upper_active_set = 0;
    double max_kkt_violation = 0.0;

    feature_index->clear();

    for (size_t i = 0; i < x.size(); ++i) {
      if (shrink[i] >= shrinking_size) continue;

      ++active_set;
      std::fill(expected.begin(), expected.end(), 0.0);
      double cost_diff = x[i]->collins(&expected[0]);
      int error_num = x[i]->eval();
      err += error_num;
      if (error_num) ++zeroone;

      if (error_num == 0) {
        ++shrink[i];
      } else {
        shrink[i] = 0;
        double s = 0.0;
        for (size_t k = 0; k < expected.size(); ++k)
          s += expected[k] * expected[k];

        double mu = _max(0.0, (error_num - cost_diff) / s);

        if (upper_bound[i] + mu > C) {
          mu = C - upper_bound[i];
          ++upper_active_set;
        } else {
          max_kkt_violation = _max(error_num - cost_diff,
                                   max_kkt_violation);
        }

        if (mu > 1e-10) {
          upper_bound[i] += mu;
          upper_bound[i] = _min(C, upper_bound[i]);
          for (size_t k = 0; k < expected.size(); ++k)
            alpha[k] += mu * expected[k];
        }
      }
    }

    double obj = 0.0;
    for (size_t i = 0; i < feature_index->size(); ++i)
      obj += alpha[i] * alpha[i];

    std::cout << "iter="  << itr
              << " terr=" << 1.0 * err / all
              << " serr=" << 1.0 * zeroone / x.size()
              << " act=" <<  active_set
              << " uact=" << upper_active_set
              << " obj=" << obj
              << " kkt=" << max_kkt_violation << std::endl;

    if (max_kkt_violation <= 0.0) {
      std::fill(shrink.begin(), shrink.end(), 0);
      converge++;
    } else {
      converge = 0;
    }

    if (itr > maxitr || converge == 2)  break;  // 2 is ad-hoc
  }

  return true;
}

void set_default_kernel(KERNEL* kernel)
{
	kernel->type = 0;
	kernel->gamma = 1.0;
	kernel->s = 1.0;
	kernel->r = 0.0;
	kernel->d = 1;
}

void set_default_param(LEARN_PARAM* param)
{
	param->C = 1.0;
	param->cache_size = 400;
	param->nc  = -1;
	param->nsp = 20;
	param->epsilon   = 0.001;
	param->epsilonuse   = 0.001;
	param->epsilon0  = -1;
	param->projector = 0;
	param->tail_one  = 1;
	param->check_point = 0;
	param->time_limit  = 0;
	param->max_iteration = 100000;
	param->epsilon00    = -1;
	param->epsilon0use  = param->epsilon00;
	param->epsilon_alga = 1.0e-10;
	param->init_threshold = INF;
	param->epsilon_sv     = 1.0e-10;
	param->ratio = 0.1;
	param->use_init = 0;
	param->subsolver = 0; /* 0: PG 1: IPM 2: Combined Solver */
	param->steps_to_alter = 15;
	param->ws_selector = 0;
	
	// set default newin size if no specified
	if(param->nc < 0) {
		param->nc = param->nsp / 2;
	}
	if(param->epsilon0 < 0) {
		param->epsilon0 = param->epsilon;
	}
	if(param->epsilon00 < 0) {
		param->epsilon00 = param->epsilon0;
	}
	param->epsilon0use = param->epsilon00;
}

bool runSVM_MC(const std::vector<TaggerImpl* > &x,
            EncoderFeatureIndex *feature_index,
            double *alpha,
            size_t maxitr,
            float C,
            double eta,
            unsigned short shrinking_size,
            unsigned short thread_num,
            bool orthant) {
	
	McSample data(feature_index, x);
	
	LEARN_PARAM param;
	KERNEL kernel;

	set_default_param(&param);
	set_default_kernel(&kernel);
	
	CACHE cache;
	int ccol = cache_init(&cache, param.cache_size, data.m);
	{
		printf("Initialize cache complete with maximum columns: %d. \n", ccol);
	}
	
	kernel_init(&kernel,&data,&cache);
	{
		printf("Initialize kernel complete. \n");
	}
	
	svm_learn(&data, &kernel, &param, NULL, alpha);
	return true;
}

bool runCRF(const std::vector<TaggerImpl* > &x,
            EncoderFeatureIndex *feature_index,
            double *alpha,
            size_t maxitr,
            float C,
            double eta,
            unsigned short shrinking_size,
            unsigned short thread_num,
            bool orthant) {
  double old_obj = 1e+37;
  int    converge = 0;
  LBFGS lbfgs;
  std::vector<CRFEncoderThread> thread(thread_num);

  for (size_t i = 0; i < thread_num; i++) {
    thread[i].start_i = i;
    thread[i].size = x.size();
    thread[i].thread_num = thread_num;
    thread[i].x = const_cast<TaggerImpl **>(&x[0]);
    thread[i].expected.resize(feature_index->size());
  }

  size_t all = 0;
  for (size_t i = 0; i < x.size(); ++i)  all += x[i]->size();

  for (size_t itr = 0; itr < maxitr; ++itr) {
    feature_index->clear();

    for (size_t i = 0; i < thread_num; ++i) thread[i].start();
    for (size_t i = 0; i < thread_num; ++i) thread[i].join();

    for (size_t i = 1; i < thread_num; ++i) {
      thread[0].obj += thread[i].obj;
      thread[0].err += thread[i].err;
      thread[0].zeroone += thread[i].zeroone;
    }

    for (size_t i = 1; i < thread_num; ++i) {
      for (size_t k = 0; k < feature_index->size(); ++k)
        thread[0].expected[k] += thread[i].expected[k];
    }

    // update the objective and gradient for the regularization term
    size_t num_nonzero = 0;
    if (orthant) {   // L1
      for (size_t k = 0; k < feature_index->size(); ++k) {
        thread[0].obj += std::abs(alpha[k] / C);
        if (alpha[k] != 0.0) ++num_nonzero; // L1 norm results in sparse weight vector, count the nnz
      }
    } else {
      num_nonzero = feature_index->size();
      for (size_t k = 0; k < feature_index->size(); ++k) {
        thread[0].obj += (alpha[k] * alpha[k] /(2.0 * C));
        thread[0].expected[k] += alpha[k] / C;
      }
    }

    double diff = (itr == 0 ? 1.0 :
                   std::abs(old_obj - thread[0].obj)/old_obj);
    std::cout << "iter="  << itr
              << " terr=" << 1.0 * thread[0].err / all
              << " serr=" << 1.0 * thread[0].zeroone / x.size()
              << " act=" << num_nonzero
              << " obj=" << thread[0].obj
              << " diff="  << diff << std::endl;
    old_obj = thread[0].obj;

    // stop if diff < eta for 3 times, conservative terminating rule
    if (diff < eta)
      converge++;
    else
      converge = 0;

    if (itr > maxitr || converge == 3)  break;  // 3 is ad-hoc

    if (lbfgs.optimize(feature_index->size(), // size of optimal variable
                       &alpha[0], // current optimal variable
                       thread[0].obj, // objective
                       &thread[0].expected[0], // gradient at current point
                       orthant, // L1 or L2 norm
                       C // regularization term for Gaussian prior
                     ) <= 0)
      return false;
  } // end main loop

  return true;
}

bool Encoder::convert(const char* textfilename,
                      const char *binaryfilename) {
  EncoderFeatureIndex feature_index(1);
  CHECK_FALSE(feature_index.convert(textfilename, binaryfilename))
      << feature_index.what();

  return true;
}

bool Encoder::learn(const char *templfile,
                    const char *trainfile,
                    const char *modelfile,
                    bool textmodelfile, bool dump,
                    size_t maxitr,
                    size_t freq,
                    double eta,
                    double C,
                    unsigned short thread_num,
                    unsigned short shrinking_size,
                    int algorithm) {
  std::cout << COPYRIGHT << std::endl;

  CHECK_FALSE(eta > 0.0) << "eta must be > 0.0";
  CHECK_FALSE(C >= 0.0) << "C must be >= 0.0";
  CHECK_FALSE(shrinking_size >= 1) << "shrinking-size must be >= 1";
  CHECK_FALSE(thread_num > 0) << "thread must be > 0";

#ifndef CRFPP_USE_THREAD
  CHECK_FALSE(thread_num == 1)
      << "This architecture doesn't support multi-thrading";
#endif

  CHECK_FALSE(algorithm == CRF_L2 || algorithm == CRF_L1 ||
              (algorithm == MIRA && thread_num == 1) || algorithm == SVM_MC)
      <<  "MIRA doesn't support multi-thrading";

  EncoderFeatureIndex feature_index(thread_num);
  std::vector<TaggerImpl* > x;

  std::cout.setf(std::ios::fixed, std::ios::floatfield);
  std::cout.precision(5);

#define WHAT_ERROR(msg) do {                                    \
    for (std::vector<TaggerImpl *>::iterator it = x.begin();    \
         it != x.end(); ++it)                                   \
      delete *it;                                               \
    std::cerr << msg << std::endl;                              \
    return false; } while (0)

  CHECK_FALSE(feature_index.open(templfile, trainfile))
      << feature_index.what();

  {
    progress_timer pg;

    std::ifstream ifs(trainfile);
    CHECK_FALSE(ifs) << "cannot open: " << trainfile;

    std::cout << "reading training data: " << std::flush;
    size_t line = 0;
    while (ifs) {
      TaggerImpl *_x = new TaggerImpl();
      _x->open(&feature_index);
      if (!_x->read(&ifs) || !_x->shrink())
        WHAT_ERROR(_x->what());

      if (!_x->empty())
        x.push_back(_x);
      else
        delete _x;

      _x->set_thread_id(line % thread_num);

      if (++line % 100 == 0) std::cout << line << ".. " << std::flush;
    }

    ifs.close();
    std::cout << "\nDone!";
  }

  feature_index.shrink(freq);

  std::vector <double> alpha(feature_index.size());           // parameter
  std::fill(alpha.begin(), alpha.end(), 0.0);
  feature_index.set_alpha(&alpha[0]);

  std::cout << "Number of sentences: " << x.size() << std::endl;
  std::cout << "Number of features:  " << feature_index.size() << std::endl;
  std::cout << "Number of thread(s): " << thread_num << std::endl;
  std::cout << "Freq:                " << freq << std::endl;
  std::cout << "eta:                 " << eta << std::endl;
  std::cout << "C:                   " << C << std::endl;
  std::cout << "dump:                " << dump << std::endl;
  std::cout << "shrinking size:      " << shrinking_size
            << std::endl;

  progress_timer pg;
  // const int* fv = x[0]->emission_vector(0, 0);
  
  switch (algorithm) {
    case MIRA:
      if (!runMIRA(x, &feature_index, &alpha[0],
                   maxitr, C, eta, shrinking_size, thread_num))
          WHAT_ERROR("MIRA execute error");
      break;
    case CRF_L2:
      if (!runCRF(x, &feature_index, &alpha[0],
                  maxitr, C, eta, shrinking_size, thread_num, false))
           WHAT_ERROR("CRF_L2 execute error");
      break;
    case CRF_L1:
      if (!runCRF(x, &feature_index, &alpha[0],
                  maxitr, C, eta, shrinking_size, thread_num, true))
           WHAT_ERROR("CRF_L1 execute error");
      break;
    case SVM_MC:
      if (!runSVM_MC(x, &feature_index, &alpha[0],
    	                  maxitr, C, eta, shrinking_size, thread_num, true))
    	   WHAT_ERROR("SVM_MC execute error");
      break;
  }

  if (!feature_index.save(modelfile, textmodelfile))
    WHAT_ERROR(feature_index.what());

  std::cout << "\nDone!";

  // const int* fv = x[0]->emission_vector(0, 0);
  if(dump) {
	  std::string dumpfile(trainfile);
	  dumpfile += ".feature";

	  std::cout << "\nDump feature to '" << dumpfile << "' ...";

	  std::ofstream ofs(dumpfile.c_str());
	  CHECK_FALSE(ofs) << "cannot open: " << dumpfile;
	  std::vector<TaggerImpl*>::iterator it;
	  std::map<int, int> fmap;
	  unsigned int n=1;
	  for(it=x.begin(); it!=x.end(); ++it) {
		  TaggerImpl* tag = *it;
		  unsigned int i,j;
		  for(i=0; i < tag->size(); ++i) {
			  fmap.clear();
			  for(j=0; j < tag->ysize(); ++j) {
				  const int *flist = tag->emission_vector(i,j);
				  unsigned int k=0;
				  while(flist[k] >= 0) {
					  fmap[flist[k]]+=1;
					  ++k;
				  }
			  }
			  std::map<int, int>::iterator it2 = fmap.begin();
			  ofs << n << " " << tag->y(i);
			  for(; it2!=fmap.end(); ++it2) {
				  ofs << " " << it2->first << ":" << it2->second;
			  }
			  ofs << std::endl;
		  }
		  ++n;
	  }
	  ofs.close();
	  std::cout << "\nDump feature done!\n";
  }

  for (std::vector<TaggerImpl *>::iterator it = x.begin();
       it != x.end(); ++it)
    delete *it;
  return true;
}
}

int crfpp_learn(int argc, char **argv) {
  static const CRFPP::Option long_options[] = {
    {"freq",     'f', "1",      "INT",
     "use features that occuer no less than INT(default 1)" },
    {"maxiter" , 'm', "100000", "INT",
     "set INT for max iterations in LBFGS routine(default 10k)" },
    {"cost",     'c', "1.0",    "FLOAT",
     "set FLOAT for cost parameter(default 1.0)" },
    {"eta",      'e', "0.0001", "FLOAT",
     "set FLOAT for termination criterion(default 0.0001)" },
    {"convert",  'C',  0,       0,
     "convert text model to binary model" },
    {"textmodel", 't', 0,       0,
     "build also text model file for debugging" },
    {"algorithm",  'a', "CRF",   "(CRF|MIRA)", "select training algorithm" },
    {"thread", 'p',   "1",       "INT",   "number of threads(default 1)" },
    {"dump", 'd',   0,       0,   "dump features(default 0)" },
    {"shrinking-size", 'H', "20", "INT",
     "set INT for number of iterations variable needs to "
     " be optimal before considered for shrinking. (default 20)" },
    {"version",  'v', 0,        0,       "show the version and exit" },
    {"help",     'h', 0,        0,       "show this help and exit" },
    {0, 0, 0, 0, 0}
  };

  CRFPP::Param param;

  param.open(argc, argv, long_options);

  if (!param.help_version()) return 0;

  const bool convert = param.get<bool>("convert");

  const std::vector<std::string> &rest = param.rest_args();
  if (param.get<bool>("help") ||
      (convert && rest.size() != 2) || (!convert && rest.size() != 3)) {
    std::cout << param.help();
    return 0;
  }

  const size_t         freq           = param.get<int>("freq");
  const size_t         maxiter        = param.get<int>("maxiter");
  const double         C              = param.get<float>("cost");
  const double         eta            = param.get<float>("eta");
  const bool           textmodel      = param.get<bool>("textmodel");
  const bool           dump           = param.get<bool>("dump");
  const unsigned short thread         = param.get<unsigned short>("thread");
  const unsigned short shrinking_size = param.get<unsigned short>("shrinking-size");
  std::string salgo = param.get<std::string>("algorithm");

  toLower(&salgo);

  int algorithm = CRFPP::Encoder::MIRA;
  if (salgo == "crf" || salgo == "crf-l2") {
    algorithm = CRFPP::Encoder::CRF_L2;
  } else if (salgo == "crf-l1") {
    algorithm = CRFPP::Encoder::CRF_L1;
  } else if (salgo == "mira") {
    algorithm = CRFPP::Encoder::MIRA;
  } else if(salgo == "svmmc") {
	  algorithm = CRFPP::Encoder::SVM_MC;
  } else if(salgo == "svmhmm") {
	  algorithm = CRFPP::Encoder::SVM_HMM;
  } else {
    std::cerr << "unknown alogrithm: " << salgo << std::endl;
    return -1;
  }

  CRFPP::Encoder encoder;
  if (convert) {
    if (!encoder.convert(rest[0].c_str(), rest[1].c_str())) {
      std::cerr << encoder.what() << std::endl;
      return -1;
    }
  } else {
    if (!encoder.learn(rest[0].c_str(),
                       rest[1].c_str(),
                       rest[2].c_str(),
                       textmodel, dump,
                       maxiter, freq, eta, C, thread, shrinking_size,
                       algorithm)) {
      std::cerr << encoder.what() << std::endl;
      return -1;
    }
  }

  return 0;
}

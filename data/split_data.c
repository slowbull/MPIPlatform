#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL
#include<armadillo>
#include<vector>
#include <sys/stat.h>
#include <sys/types.h>


using namespace arma;

struct feature_node
{
	int index;
	double value;
};

struct problem
{
	int l, n;
	double *y;
	struct feature_node **x;
	double bias;            /* < 0 if no bias term */  
};

struct parameter
{
	int solver_type;

	/* these are for training only */
	double eps;	        /* stopping criteria */
	double C;
	int nr_weight;
	int *weight_label;
	double* weight;
	double p;
	double *init_sol;
};

struct model
{
	struct parameter param;
	int nr_class;		/* number of classes */
	int nr_feature;
	double *w;
	int *label;		/* label of each class */
	double bias;
};

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}


static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void read_problem(const char *filename);

struct feature_node *x_space;
struct parameter param;
struct problem prob;
struct model* model_;
int flag_cross_validation;
int flag_find_C;
int flag_C_specified;
int flag_solver_specified;
int nr_fold;
double bias;
int workers;
double test_ratio;

int main(int argc, char **argv)
{

	if(argc!=4){
		printf("Read data from libsvm data, and partition and save in distributed armadillo data files. \n");
		printf("usage of this file: ./split_data /pathto/data num_partitions test_ratio\n");
		return 0;
	}
	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;

	strcpy(input_file_name, argv[1]);
	workers = std::strtol(argv[2],NULL,0);
	test_ratio = std::strtod(argv[3],NULL);
	read_problem(input_file_name);

	free(prob.y);
	free(prob.x);
	free(x_space);
	free(line);

	return 0;
}

// read in a problem (in libsvm format)
void read_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	size_t elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;
	bias = -1;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;
	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
		elements++; // for bias term
		prob.l++;
	}
	rewind(fp);

	prob.bias=bias;


	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct feature_node *,prob.l);
	x_space = Malloc(struct feature_node,elements+prob.l);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);

			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;

		if(prob.bias >= 0)
			x_space[j++].value = prob.bias;

		x_space[j++].index = -1;
	}

	if(prob.bias >= 0)
	{
		prob.n=max_index+1;
		for(i=1;i<prob.l;i++)
			(prob.x[i]-2)->index = prob.n;
		x_space[j-2].index = prob.n;
	}
	else
		prob.n=max_index;

	printf("l: %d n:%d ele:%d \n", prob.l, prob.n, (int)elements);


	 // shuffle vector
	std::vector<int> randperm(prob.l,0);
	for(int i=0; i<prob.l; i++){
		randperm[i] = i;
	}
	std::random_shuffle(randperm.begin(), randperm.end());

	int ii=0;
	int count=0;
	std::vector<int> tmp_loc;
	std::vector<double> tmp_val;
	for(i=0; i<elements+prob.l; i++){
		if(x_space[i].index==-1){
			ii++;
			if(ii>=prob.l) break;	
	   	 }
		else{
			tmp_loc.push_back(x_space[i].index-1);
			tmp_loc.push_back(randperm[ii]);
			//tmp_loc.push_back(ii);
			tmp_val.push_back(x_space[i].value);
			count++;
		}
	}
	
	umat my_loc = conv_to<Mat<uword> >::from(tmp_loc);
	std::vector<int>().swap(tmp_loc); // release memory
	my_loc.reshape(2,count);
	vec my_val = conv_to<vec>::from(tmp_val);
	std::vector<double>().swap(tmp_val); // release memory
    sp_mat my_x(my_loc,my_val); // n*l because sp_mat in armadillo is based on CSC.
	

	std::vector<double> vec_y(prob.y,prob.y+prob.l);
	std::vector<int> tmp_loc_y;
	std::vector<double> tmp_val_y;
	for(i=0; i<prob.l; i++){
		tmp_loc_y.push_back(0);
		tmp_loc_y.push_back(randperm[i]);
		tmp_val_y.push_back(vec_y[i]);
	}

	umat my_loc_y = conv_to<Mat<uword> >::from(tmp_loc_y);
	std::vector<int>().swap(tmp_loc_y); // release memory
	my_loc_y.reshape(2,prob.l);
	vec my_val_y = conv_to<vec>::from(tmp_val_y);
	std::vector<double>().swap(tmp_val_y); // release memory
    sp_mat my_y(my_loc_y,my_val_y); // n*l because sp_mat in armadillo is based on CSC.
	my_y = my_y.t();
	

	int num = my_y.n_rows;
	int train_num = int (num * (1 - test_ratio));
	int test_num = num - train_num;

	printf("num of data %d \n", num);
	int train_data_per_node = std::ceil(1.0*train_num/workers);
	int test_data_per_node = std::ceil(1.0*test_num/workers);
	
	char dirname[200];
	std::strcpy(dirname, filename);
	std::strcat(dirname, "_split");
	printf(" worker %s\n", dirname);
	mkdir(dirname, 0777);

	for(int i=1; i<=workers; i++){
		int left = (i-1)*train_data_per_node;
		int right = i*train_data_per_node > train_num ? train_num : i*train_data_per_node;
		sp_mat tmp_images = my_x.cols(left, right-1); 
		sp_mat tmp_labels = my_y.rows(left, right-1); 
		printf(" worker %d  num of training data %d\n", i, right-left);
		char numstr[21];
		sprintf(numstr, "%d", i);
		char tmp_file_feas[200];
		std::strcpy(tmp_file_feas, dirname);
		std::strcat(tmp_file_feas, "/features.mat_");
		std::strcat(tmp_file_feas, numstr);
		char tmp_file_labs[200];
		std::strcpy(tmp_file_labs, dirname);
		std::strcat(tmp_file_labs, "/labels.mat_");
		std::strcat(tmp_file_labs, numstr);
		tmp_images.save(tmp_file_feas, arma_binary);
		tmp_labels.save(tmp_file_labs, arma_binary);
	}
	
	if(test_ratio != 0){ 
	for(int i=1; i<=workers; i++){
		int left = (i-1)*test_data_per_node;
		int right = i*test_data_per_node > test_num ? test_num : i*test_data_per_node;
		sp_mat tmp_images = my_x.cols(train_num+left, train_num+right-1); 
		sp_mat tmp_labels = my_y.rows(train_num+left, train_num+right-1); 
		printf(" worker %d  num of testing data %d\n", i, right-left);
		char numstr[21];
		sprintf(numstr, "%d", i);
		char tmp_file_feas[200];
		std::strcpy(tmp_file_feas, dirname);
		std::strcat(tmp_file_feas, "/test_features.mat_");
		std::strcat(tmp_file_feas, numstr);
		char tmp_file_labs[200];
		std::strcpy(tmp_file_labs, dirname);
		std::strcat(tmp_file_labs, "/test_labels.mat_");
		std::strcat(tmp_file_labs, numstr);
		tmp_images.save(tmp_file_feas, arma_binary);
		tmp_labels.save(tmp_file_labs, arma_binary);
	}
	}

    printf("Saving data done \n");

	fclose(fp);

}

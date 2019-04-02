#include <iostream>
#include <iomanip>
#include <fstream>
#include <armadillo>
#include <math.h>
#include <stdio.h>

//#include "main.h"

using namespace std;
using namespace arma;


namespace vmc
{

    int dimensions;
    int particles;

    int metropolis_steps;
    int alpha_steps;
    
    double sum_El;
    double sum_El2;
    double sum_El_psi_alpha;
    double sum_1_psi_alpha;

    double metropolis_acceptance_rate;

    double alpha;
    double beta;
    double omega;

    double DR;

    std::function<double(const mat&)> wave_func;
    //std::function<double(mat&)> wave_func;
    std::function<double(const mat&)> local_E ;
    std::function<double(double, double, double, double)> run_metropolis ;
    std::function<double(const mat&)> d_psi_d_alpha;
}

namespace ran
{
    // Usage in code:
    //double a = ran::normal(ran::generator)
    //double b = ran::uniform(ran::generator) 
    
    std::mt19937_64 generator (randu());
    std::uniform_real_distribution<double> uniform(0, 1);
    std::normal_distribution<double> normal(0.0, 1.0);
}


mat initialize();
double run_metropolis_importance_sampling(double dt, double dr, double learning_rate, double diffusion);
double run_metropolis_brute_force(double dt, double dr, double learning_rate, double diffusion);

double wave_function_3D(const mat& r);

double wave_function_interaction(mat& r);

double local_energy_analytical(const mat& r);
double local_energy_analytical_interaction(const mat& r);

rowvec drift_force(rowvec r);
rowvec drift_force_interaction(mat r);

double greens_parameter(const mat& r_new, const mat& r_old, double diffussion, double dt);

double d_psi_d_alpha_3D(const mat& position);
double expectation_E(double sum_local_energy);
double expectation_El_psi_alpha(double sum_El_psi_d_psi_d_alpha);
double expectation_1_psi_alpha(double sum_1_psi_d_psi_d_alpha);

double numerical_derivative_wave_function(const mat& r, double dr);
double local_energy_numerical(const mat& r, double dr);

double metropolis_importance_sampling(double dt, double dr, double learning_rate, double diffusion);
double metropolis_brute_force(double dt, double dr, double learning_rate, double diffusion);
//void write_to_file(string data);

void averages();
void reset_values();


int main()
{
    std::cout << "Beginning program"  << std::endl;

    vmc::dimensions = 3;
    vmc::particles = 16;

    vmc::metropolis_steps = pow(2,16);
    vmc::alpha_steps = 30;
    
    // double dt = 0.001;

    vmc::beta = 1.0;
    vmc::alpha = 0.3; 
    vmc::omega = 1.0;

    double learning_rate = 0.07;
    double diffusion = 0.5;
    double dt = 0.001;
    double dr = 0.003;
    vmc::DR = 0.003;



    // std::function<const mat&, double, double> wave_function;
    // wave_function = wave_function_3D
    // foo(std::function<const mat&, double, double> wave_function) {
    //      return wavefunction(...);
    // }

    vmc::d_psi_d_alpha = d_psi_d_alpha_3D;
    vmc::wave_func = wave_function_3D;
    vmc::local_E = local_energy_analytical_interaction;
    vmc::run_metropolis = run_metropolis_importance_sampling;
    vmc::run_metropolis(dt, dr, learning_rate, diffusion);

    return 0;
}


double run_metropolis_importance_sampling(double dt, double dr, double learning_rate, double diffusion)
{
    double gradients_for_alpha = 0;
    metropolis_importance_sampling(dt, dr, learning_rate, diffusion);
    averages();
    for (int i=0 ; i<vmc::alpha_steps ; i++ ){
        gradients_for_alpha = 2* (vmc::sum_El_psi_alpha/vmc::metropolis_steps) - 2*(vmc::sum_El/vmc::metropolis_steps) * (vmc::sum_1_psi_alpha/vmc::metropolis_steps);
        cout << gradients_for_alpha << " alpha gradients " << endl;
        vmc::alpha = vmc::alpha - learning_rate * gradients_for_alpha;
        reset_values();
        metropolis_importance_sampling(dt, dr, learning_rate, diffusion);
        averages();
    }
    return 0;
}


double metropolis_importance_sampling(double dt, double dr, double learning_rate, double diffusion)
{

    reset_values();
    vmc::sum_El = 0;
    vmc::sum_El2 = 0;
    vmc::sum_El_psi_alpha = 0;
    vmc::sum_1_psi_alpha = 0;

    mat position = initialize();
    double local_energy;
    double wave_function;

    double new_local_energy; 
    double new_wave_function;
    mat new_position (vmc::particles, vmc::dimensions);
    
    double metropolis_acceptance = 0;
    double metropolis_steps_counter = 0;
    double acceptance_threshold = 0;
    double greens = 0;
    double a = 0.1;

    uniform_int_distribution<unsigned int> particle_dist(0, vmc::particles-1);

    // Initial fill of local energy and wave function.
    for (int k = 0 ; k < vmc::particles; k++){
        local_energy = vmc::local_E(position);
        //local_energy(k) = local_energy_numerical(position.row(k), dr);
        //wave_function = vmc::wave_func(position);
        wave_function = wave_function_interaction(position);
    }
    // Loop metropolis steps.
    new_position = position;
    for (int i = 0; i < vmc::metropolis_steps;  i++) {
        int j = particle_dist(ran::generator); // particle!



            rowvec quantum_force = drift_force(position.row(j));
            for (int k = 0 ; k < vmc::dimensions ; k ++){
                new_position(j,k) = position(j,k) + diffusion*quantum_force(k)*dt + ran::normal(ran::generator)*sqrt(dt);
            }
            new_local_energy = vmc::local_E(new_position);
            //new_wave_function = vmc::wave_func(new_position);
            new_wave_function = wave_function_interaction(new_position);
            greens = greens_parameter(new_position, position, diffusion, dt);

            acceptance_threshold = greens*new_wave_function*new_wave_function/(wave_function*wave_function);
            if (acceptance_threshold >= ran::uniform(ran::generator) ){
                position.row(j) = new_position.row(j);
                wave_function = new_wave_function;
                local_energy = new_local_energy;
                metropolis_acceptance++;
            } else {
                new_position.row(j) = position.row(j);
            }

            vmc::sum_El += local_energy;
            vmc::sum_El2 += local_energy*local_energy;
            vmc::sum_El_psi_alpha += local_energy*vmc::d_psi_d_alpha(position.row(j));
            vmc::sum_1_psi_alpha += vmc::d_psi_d_alpha(position.row(j));

            metropolis_steps_counter = metropolis_steps_counter + 1;
    }
    vmc::metropolis_acceptance_rate = metropolis_acceptance/metropolis_steps_counter;
    return 0;
}

mat initialize()
{
    mat particle_positions (vmc::particles, vmc::dimensions, fill::zeros);
    for ( int i = 0; i < vmc::particles ; i++){
        for (int j = 0; j < vmc::dimensions; j++ ){
            particle_positions(i,j) = ran::uniform(ran::generator)-0.5;
        }
    }
    return particle_positions;
}

rowvec drift_force_interaction(mat r){
    rowvec force (vmc::dimensions, fill::ones);
    for (int i = 0 ; i < vmc::particles ; i++){
        for (int l = 0 ; l < vmc::dimensions ; l ++){
            mat R_plus = r;
            mat R_minus = r;
            mat R = r;
            R_plus(i,l) += vmc::DR;
            R_minus(i,l) -= vmc::DR;
            force(l) *= (wave_function_3D(R_plus) + wave_function_3D(R_minus) - 2*wave_function_3D(R))/vmc::DR;
        }
    }
    force *= 2/wave_function_3D(r);
    return force;
}


double wave_function_3D(const mat& r)
{
    double wf = 0.;
    for (int l = 0 ; l < vmc::particles ; l++){
        wf += r(l,0)*r(l,0) + r(l,1)*r(l,1) + vmc::beta*r(l,2)*r(l,2);
    }
    return exp(-vmc::alpha*wf);
}




double local_energy_analytical(const mat& r)
{   
    double total_local_energy = 0;
    double r_2_sum;
    for (int l = 0; l < vmc::particles ; l++ ){
        r_2_sum = 0;
        for (int k = 0 ; k < vmc::dimensions ; k++){
            r_2_sum += r(l,k)*r(l,k);
        }
        total_local_energy += vmc::dimensions*vmc::alpha - 2*vmc::alpha*vmc::alpha*r_2_sum + 0.5*vmc::omega*r_2_sum;
    } 
    return total_local_energy;
} 

double local_energy_analytical_interaction(const mat& r)
{   
    double total_local_energy = 0;
    double r_2_sum;
    double a = 0.1;
    double I = 0;
    double II = 0;
    double III = 0;
    double IV = 0;

    for (int l = 0; l < vmc::particles ; l++ ){
        r_2_sum = 0;
        double I = -vmc::alpha*2 + 4*vmc::alpha*vmc::alpha*4*(r(l,0)*r(l,0) + r(l,1)*r(l,1) +r(l,2)*r(l,2));
        rowvec II_vec = -2*vmc::alpha*r.row(2);
            
        for (int i = 0 ; i < vmc::particles ; i++){
            if (i == l){
                II += 0;
            } else{
            double r_li = norm(r.row(l)-r.row(i));
            II += dot(II_vec, (r.row(l)-r.row(i))/r_li * a/(r_li*(r_li-a)));
                
            for (int j = 0; j < vmc::particles ; j++){
                double r_lj = norm(r.row(l)-r.row(j));
                if (j==l){
                    III += 0;
                } else if (j == i){
                    III += 0;
                } else {
                    III += a*a*dot(r.row(l), r.row(j))/(r_lj*r_lj*r_li*r_li*(r_lj - a)*(r_li-a));
                }
            }
            IV += (2*a*r_li - 2*a*a + a*a - 2*a*r_li)/(r_li*r_li*(r_li-a)*(r_li -a));
            }
        }
    } 
    return I + II + III + IV ;
} 

rowvec drift_force(rowvec r){
    return -4*vmc::alpha*r;
}

double greens_parameter(const mat& r_new, const mat& r_old, double diffusion, double dt) {
    double greens = 0;

    for (int i=0; i<vmc::particles; ++i) {
        double fraction = 0.25/diffusion/dt;

        rowvec g_xy_vec = (r_new(i) - r_old(i) - diffusion*dt*drift_force(r_new.row(i)));
        double g_xy = dot(g_xy_vec, g_xy_vec);

        rowvec g_yx_vec = (r_old(i) - r_new(i) - diffusion*dt*drift_force(r_old.row(i)));
        double g_yx = dot(g_yx_vec, g_yx_vec);

        greens += exp(fraction * (- g_xy + g_yx));
    }
    return greens;
}


double d_psi_d_alpha_3D(const mat& position){
    return -(position(0)*position(0) + position(1)*position(1) + position(2)*position(2));
}


double wave_function_interaction(mat& r)
{
    double total_wave_function = 1;
    double r_ij;
    //mat R = r;
    
    double a = 0.1;

    total_wave_function *= vmc::wave_func(r);
    for (int i = 0; i < vmc::particles ; i++ ){ // Den ene partikkelen
        for (int j = i+1 ; j < vmc::particles ; j++ ){
                r_ij = sqrt((r(i,0)-r(j,0))*(r(i,0)-r(j,0)) + (r(i,1)-r(j,1))*(r(i,1)-r(j,1)) + (r(i,2)-r(j,2))*(r(i,2)-r(j,2)));
            if (r_ij > a){
                total_wave_function *= (1-a/r_ij);
            } else {
                total_wave_function *= 0;
            }
        }
    } 
    return total_wave_function;
} 




void averages(){
    double expectance_E = vmc::sum_El/vmc::metropolis_steps;
    double expectance_E_squared = vmc::sum_El2/vmc::metropolis_steps;
    double variance = expectance_E_squared-expectance_E*expectance_E ;
    double STD = variance/sqrt(vmc::metropolis_steps);

    cout << " \n----------------------------" << endl;
    /*cout << "Final wave function: ", self.final_wave_function  << endl;
    cout << "Final local energy : ", self.final_local_energy  << endl;
    cout << "Final position     : ", self.final_position  << endl; */
    cout << "Alpha              : " << vmc::alpha << endl;
    cout << "Acceptance rate    : " << vmc::metropolis_acceptance_rate << endl;
    cout << "<E>                : " << expectance_E  << endl;
    cout << "<E^2>              : " << expectance_E_squared  << endl;
    cout << "E variance         : " << variance  << endl;
    cout << "STD                : " << STD  << endl;
        
    cout << "---------------------------- \n " << endl;
}

void reset_values()
{
    vmc::sum_El = 0;
    vmc::sum_El2 = 0;
    vmc::sum_El_psi_alpha = 0;
    vmc::sum_1_psi_alpha = 0;
    vmc::metropolis_acceptance_rate = 0;
}


double numerical_derivative_wave_function(const mat& r, double dr){
    return (vmc::wave_func(r+dr) + vmc::wave_func(r-dr) - 2*vmc::wave_func(r))/(dr*dr);
}

double local_energy_numerical(const mat& r, double dr){
    return numerical_derivative_wave_function(r, dr)/vmc::wave_func(r);
}


double run_metropolis_brute_force(double dt, double dr, double learning_rate, double diffusion)
{
    double gradients_for_alpha = 0;
    metropolis_brute_force(dt, dr, learning_rate, diffusion);
    averages();
    for (int i=0 ; i<vmc::alpha_steps ; i++ ){
        gradients_for_alpha = 2* (vmc::sum_El_psi_alpha/vmc::metropolis_steps) - 2*(vmc::sum_El/vmc::metropolis_steps) * (vmc::sum_1_psi_alpha/vmc::metropolis_steps);
        vmc::alpha = vmc::alpha - learning_rate * gradients_for_alpha;
        reset_values();
        metropolis_brute_force(dt, dr, learning_rate, diffusion);
        averages();
    }
    return 0;
}
  

double metropolis_brute_force(double dt, double dr, double learning_rate, double diffusion)
{

    reset_values();

    mat position = initialize();
    double local_energy;
    double wave_function; 

    double new_local_energy;
    double new_wave_function;
    mat new_position (vmc::particles, vmc::dimensions);
    
    double metropolis_acceptance = 0;
    double metropolis_steps_counter = 0;
    double greens = 0;
    double a = 0.1;

    uniform_int_distribution<unsigned int> particle_dist(0, vmc::particles-1);

    // Initial fill of local energy and wave function.
    for (int k = 0 ; k < vmc::particles; k++){
        local_energy = vmc::local_E(position);
        //local_energy(k) = local_energy_numerical(position.row(k), dr);
        wave_function = vmc::wave_func(position);
        //wave_function(k) = wave_function_interaction(position, a);
    }
    // Loop metropolis steps.
    for (int i = 0; i < vmc::metropolis_steps;  i++) {
        int j = particle_dist(ran::generator); // particle!
            rowvec quantum_force = drift_force(position.row(j));
            for (int k = 0 ; k < vmc::dimensions ; k ++){
                new_position(j,k) = position(j,k) + (ran::uniform(ran::generator) - 0.5)*dt;
            }
            /*new_position(j,0) = position(j,0) + (ran::uniform(ran::generator) - 0.5)*dt;
            new_position(j,1) = position(j,1) + (ran::uniform(ran::generator) - 0.5)*dt;
            new_position(j,2) = position(j,2) + (ran::uniform(ran::generator) - 0.5)*dt;
            */

            new_local_energy = vmc::local_E(new_position);
            //new_local_energy = local_energy_numerical(new_position.row(j), dr);
            new_wave_function = vmc::wave_func(new_position);

            double acceptance_threshold = new_wave_function*new_wave_function /(wave_function*wave_function);
            if (acceptance_threshold >= ran::uniform(ran::generator) ){ 
                position.row(j) = new_position.row(j);
                wave_function = new_wave_function;
                local_energy = new_local_energy;
                metropolis_acceptance++;
            }

            vmc::sum_El += local_energy;
            vmc::sum_El2 += local_energy*local_energy;
            vmc::sum_El_psi_alpha += local_energy*vmc::d_psi_d_alpha(position.row(j));
            vmc::sum_1_psi_alpha +=  vmc::d_psi_d_alpha(position.row(j));

            //cout << vmc::sum_El << "// " << vmc::metropolis_steps << "expec_E" << endl;

            metropolis_steps_counter = metropolis_steps_counter + 1;
    }
    vmc::metropolis_acceptance_rate = metropolis_acceptance/metropolis_steps_counter;
    return 0;
}


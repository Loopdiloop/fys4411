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

    std::function<double(const mat&)> wave_func;
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

/*
wavefuncc
derivert (alpha)
laplace
qforce kryssledd
El
*/

mat initialize();
double run_metropolis_importance_sampling(double dt, double dr, double learning_rate, double diffusion);
double run_metropolis_brute_force(double dt, double dr, double learning_rate, double diffusion);

double wave_function_1D(const mat& r);
double wave_function_2D(const mat& r);
double wave_function_3D(const mat& r);

double wave_function_interaction(const mat& r, double a);

double local_energy_analytical(const mat& r);
rowvec drift_force(rowvec r);

double greens_parameter(const mat& r_new, const mat& r_old, double diffussion, double dt);

//(double X, double Y, const mat& r_Y, const mat& r_X, const mat& r_tot, double diffusion, double dt);
double d_psi_d_alpha_1D(const mat& position);
double d_psi_d_alpha_2D(const mat& position);
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

    vmc::dimensions = 2;
    vmc::particles = 15;

    vmc::metropolis_steps = 100000;
    vmc::alpha_steps = 30;
    
    // double dt = 0.001;

    vmc::beta = 1.0;
    vmc::alpha = 0.3; 
    vmc::omega = 1.0;

    double learning_rate = 0.3;
    double diffusion = 0.5;
    double dt = 0.001;
    double dr = 0.003;

    // std::function<const mat&, double, double> wave_function;
    // wave_function = wave_function_3D
    // foo(std::function<const mat&, double, double> wave_function) {
    //      return wavefunction(...);
    // }


    if (vmc::dimensions == 1){
        vmc::wave_func = wave_function_1D;
        vmc::d_psi_d_alpha = d_psi_d_alpha_1D;
    }
    else if (vmc::dimensions == 2){
        vmc::wave_func = wave_function_2D;
        vmc::d_psi_d_alpha = d_psi_d_alpha_2D;
    }
    else if (vmc::dimensions == 3){
        vmc::wave_func = wave_function_3D;
        vmc::d_psi_d_alpha = d_psi_d_alpha_3D;
    }
    else {
        cout << "ERROR. Check dimensions! Dim = " << vmc::dimensions << endl;
        exit(0);
    

    vmc::local_E = local_energy_analytical;
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
        wave_function = vmc::wave_func(position);
        //wave_function(k) = wave_function_interaction(position, a);
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
            new_wave_function = vmc::wave_func(new_position);

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
        wave_function = vmc::wave_func(position);
    }
    // Loop metropolis steps.
    for (int i = 0; i < vmc::metropolis_steps;  i++) {
        int j = particle_dist(ran::generator); // particle!
            rowvec quantum_force = drift_force(position.row(j));
            for (int k = 0 ; k < vmc::dimensions ; k ++){
                new_position(j,k) = position(j,k) + (ran::uniform(ran::generator) - 0.5)*dt;
            }

            new_local_energy = vmc::local_E(new_position);
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



double wave_function_1D(const mat& r)
{
    double wf = 1.;
    for (int l = 0 ; l < vmc::particles ; l++){
        wf *= exp(-vmc::alpha*(r(l,0)*r(l,0)));
    }
    return wf;
} 

double wave_function_2D(const mat& r)
{

    double wf = 1.;
    for (int l = 0 ; l < vmc::particles ; l++){
        wf *= exp(-vmc::alpha*(r(l,0)*r(l,0) + r(l,1)*r(l,1)));
    }
    return wf;
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

rowvec drift_force(rowvec r)
{
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


double d_psi_d_alpha_1D(const mat& position)
{
    return -(position(0)*position(0));
}

double d_psi_d_alpha_2D(const mat& position)
{
    return -(position(0)*position(0) + position(1)*position(1));
}

double d_psi_d_alpha_3D(const mat& position)
{
    return -(position(0)*position(0) + position(1)*position(1) + position(2)*position(2));
}




double numerical_derivative_wave_function(const mat& r, double dr)
{
    """ Numericals does not work numerically yet. """;
    return (vmc::wave_func(r+dr) + vmc::wave_func(r-dr) - 2*vmc::wave_func(r))/(dr*dr);
}

double local_energy_numerical(const mat& r, double dr)
{
    """ Numericals does not work numerically yet. """;
    return numerical_derivative_wave_function(r, dr)/vmc::wave_func(r);
}

double wave_function_interaction(const mat& r, double a)
{
    """ Trenger  PI g() PI f() . Needs input r for all positions.""";
    double total_wave_function = 1;
    double r_ij;
    for (int i = 0; i < vmc::particles ; i++ ){ // Den ene partikkelen
        total_wave_function *= vmc::wave_func(r.row(i));
        for (int j = i+1 ; j < vmc::particles ; j++ ){ // Partiklene den skal interagere med
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


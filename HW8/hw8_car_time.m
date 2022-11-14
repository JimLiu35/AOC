clear;


hw8_car
u0 = out.CONTROLS(:, 2:3);


BEGIN_ACADO;                                % Always start with "BEGIN_ACADO". 
    
    acadoSet('problemname', 'hw8_car_time');
    
    % (a) Define states and controls 
    DifferentialState       Px Py theta v delta;
    Control                 ua u_delta;
    Parameter               T；
    
    
    %% (a) Differential Equation
%     t_f = 10;
    num_steps = 64;
    f = acado.DifferentialEquation(0, T);
    
    f.add(dot(Px) == v * cos(theta));
    f.add(dot(Py) == v * sin(theta));
    f.add(dot(theta) == v * tan(delta));
    f.add(dot(v) == ua);
    f.add(dot(delta) == u_delta);

    
    %% (b) Optimal Control Problem
    ocp = acado.OCP(0.0, T, num_steps);
    
    
                      
    % (b) Minimize control effort
%     u = [ua, u_delta];
    ocp.minimizeLSQ(u_delta, 0);
    ocp.minimizeMayerTerm(T);
    

    
    % (c) Path constraints
    ocp.subjectTo( f );
    ocp.subjectTo( -5 <= v <= 5);
    ocp.subjectTo( -5 <= ua <= 5);
    ocp.subjectTo( -pi/4 <= delta <= pi/4);
    ocp.subjectTo( -pi/6 <= u_delta <= pi/6);
    P = [Px, Py];
    P_os = [-7, 0];
    dist = P - P_os;
    r = 1;
    ocp.subjectTo( norm(dist) <= r );
    ocp.subjectTo( 3 <= T <= 10);

    
    % (d) Initial Conditions
    ocp.subjectTo( 'AT_START', Px == -10.0 );
    ocp.subjectTo( 'AT_START', Py == 1.0 );
    ocp.subjectTo( 'AT_START', v == 0.0 );
    ocp.subjectTo( 'AT_START', theta == 0.0 );
    ocp.subjectTo( 'AT_START', delta == 0.0 );

    
    % (d) Final boundary conditions
    ocp.subjectTo( 'AT_END', Px == 0.0 );
    ocp.subjectTo( 'AT_END', Py == 0.0 );
    ocp.subjectTo( 'AT_END', v == 0.0 );
    ocp.subjectTo( 'AT_END', theta == 0.0 );
    
    %% (e) Optimization Algorithm
    algo = acado.OptimizationAlgorithm(ocp);
    algo.set('KKT_TOLERANCE', 10e-8);
    algo.set('DISCRETIZATION_TYPE', 'MULTIPLE_SHOOTING');
    algo.set('MAX_NUM_ITERATIONS', 500);
    
    
    
    
END_ACADO;           % Always end with "END_ACADO".
                     % This will generate a file problemname_ACADO.m. 
                     % Run this file to get your results. You can
                     % run the file problemname_ACADO.m as many
                     % times as you want without having to compile again.

% Run the test
out = hw8_car_RUN();

draw;
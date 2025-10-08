%% Quadruple Tank System with MPC
clc; clear; close all;

%% System Parameters
A1 = 28; A2 = 32; A3 = 28; A4 = 32;
a1 = 0.071; a3 = 0.071; a2 = 0.057; a4 = 0.057;
kc = 0.5; g = 981; gamma1 = 0.7; gamma2 = 0.6;
k1 = 3.33; k2 = 3.35; ts = 0.1;
h0 = [12.4; 12.7; 1.8; 1.4];
u = [0; 0];

% Linearized Discrete State-Space Model
Am = [A1; A2; A3; A4];
am = [a1; a2; a3; a4];
T = (Am ./ am) .* (h0 .^ 0.5) * sqrt(2 / g);
AT = [-1/T(1) 0 A3/(A1*T(3)) 0;
      0 -1/T(2) 0 A4/(A2*T(4));
      0 0 -1/T(3) 0;
      0 0 0 -1/T(4)];
BT = [gamma1*k1/A1 0;
      0 gamma2*k2/A2;
      0 (1-gamma2)*k2/A3;
      (1-gamma1)*k1/A4 0];
C = [kc 0 0 0; 0 kc 0 0];

A = expm(AT*ts);
B = inv(AT)*(A-eye(4))*BT;
nu = size(B, 2); % Number of control inputs

%% Kalman Filter Parameters
Q = 0.001*10 * eye(4);  % Process noise covariance
R = 0.00001*2 * eye(2);   % Measurement noise covariance
P_post = (10^-2) * eye(4);  % Initial error covariance
x_hat = 0*h0;       % Initial state estimate
u = [0; 0];       % Initial control input

%% MPC Parameters
Np = 50; Nc = 3; % Prediction and control horizons
Qy = diag([10, 10, 0, 0]); % Focus on h1, h2
Qu = diag([0.01, 0.01]);
DUmin = 5 * [-1; -1]; DUmax = 5 * [1; 1];
Umin = [0; 0]; Umax = [20; 20];
setpoint = [13.4; 13.7]; % Desired h1, h2 levels

%% Simulation Parameters
n = 100; % Time steps
time = 0:ts:(n-1)*ts;

% Data storage
x_history = zeros(4, n);
x_hat_history = zeros(4, n);
u_history = zeros(2, n);

% Initial state
x = 0*h0;

%% Main Simulation Loop
for k = 1:n
    % --- Kalman Filter Estimation ---
    x_pred = A * x_hat + B * u;
    P_pred = A * P_post * A' + Q;
    y_meas = C * x + sqrt(R) * randn(2, 1);
    K = P_pred * C' / (C * P_pred * C' + R);
    x_hat = x_pred + K * (y_meas - C * x_pred);
    P_post = (eye(4) - K * C) * P_pred;

    % --- MPC Optimization ---
    [Phi, Gamma] = mpc_prediction_matrices(AT, BT, C, Np);

Gamma = compute_gamma(AT, BT, C, Np, Nc);
% Block diagonal weighting matrices
Qy = diag([1, 1]); % Weights for h1 and h2
Q_blk = kron(eye(Np), Qy);  % Np * ny x Np * ny
R_blk = kron(eye(Nc), Qu);  % Nc * nu x Nc * nu

% Hessian matrix for QP
H = 2 * (Gamma' * Q_blk * Gamma + R_blk);

% Linear term
f = 2 * Gamma' * Q_blk * Phi * x_hat;

% Constraints
A_con = [eye(Nc * nu); -eye(Nc * nu)];
b_con = [repmat(DUmax, Nc, 1); repmat(-DUmin, Nc, 1)];
    
    options = optimset('Display', 'off');
    DU_opt = quadprog(H, f, A_con, b_con, [], [], [], [], [], options);
    
    du = DU_opt(1:2);
    u = u + du;
    u = max(Umin, min(Umax, u)); % Saturation

    % --- Plant Update ---
    x = AT * x + BT * u + sqrt(Q) * randn(4, 1);

    % Store Results
    x_history(:, k) = x;
    x_hat_history(:, k) = x_hat;
    u_history(:, k) = u;
end

%% Plot Results
figure;
subplot(2, 1, 1);
plot(time, x_history(1, :)+h0(1), 'b', time, x_history(2, :)+h0(2), 'r');
hold on;
plot(time, x_hat_history(1, :)+h0(1), '--b', time, x_hat_history(2, :)+h0(2), '--r');
yline(setpoint(1), '--b', 'h1 setpoint');
yline(setpoint(2), '--r', 'h2 setpoint');
xlabel('Time (s)'); ylabel('Tank Levels (cm)');
legend('h1 (actual)', 'h2 (actual)', 'h1 (estimated)', 'h2 (estimated)');
title('Tank Levels');

subplot(2, 1, 2);
stairs(time, u_history(1, :), 'b');
hold on;
stairs(time, u_history(2, :), 'r');
xlabel('Time (s)'); ylabel('Control Input (V)');
legend('u1', 'u2');
title('Control Inputs');


function [Phi, Gamma] = mpc_prediction_matrices(A, B, C, Np)
    % Computes the prediction (Phi) and control (Gamma) matrices for MPC.
    % Inputs:
    %   A, B: State-space matrices of the plant
    %   C: Output matrix
    %   Np: Prediction horizon
    % Outputs:
    %   Phi: Prediction matrix
    %   Gamma: Control influence matrix
    
    nx = size(A, 1);  % Number of states
    nu = size(B, 2);  % Number of inputs
    ny = size(C, 1);  % Number of outputs

    % Construct the Phi matrix
    Phi = zeros(Np * ny, nx);
    A_k = eye(nx);
    for k = 1:Np
        A_k = A_k * A;  % A^k
        Phi((k-1)*ny + 1:k*ny, :) = C * A_k;
    end

    % Construct the Gamma matrix
    Gamma = zeros(Np * ny, Np * nu);
    for i = 1:Np
        for j = 1:i
            idx_row = (i-1)*ny + 1:i*ny;
            idx_col = (j-1)*nu + 1:j*nu;
            Gamma(idx_row, idx_col) = C * (A^(i-j)) * B;
        end
    end
end

function Gamma = compute_gamma(A, B, C, Np, Nc)
    % A, B, C: State-space matrices
    % Np: Prediction horizon
    % Nc: Control horizon

    [nx, nu] = size(B);  % Number of states, inputs
    ny = size(C, 1);     % Number of outputs

    Gamma = zeros(Np * ny, Nc * nu); % Preallocate Gamma

    for i = 1:Np
        for j = 1:Nc
            if i >= j
                Gamma((i-1)*ny+1:i*ny, (j-1)*nu+1:j*nu) = C * A^(i-j) * B;
            end
        end
    end
end
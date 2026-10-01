%% 4.4 算子扰动控制函数的数值验证
%
% 在两个原始数值实验中加入算子扰动 h，并按照误差模型
%       ||F_h(x)-F(x)||_Y <= h*Omega(x),
%       ||y^delta-y||_Y <= delta,
% 两个实验：
%   1) 二维逆热传导问题；
%   2) 二维位场向下延拓问题。
% 两个实验均固定数据相对误差 delta_rel = 0.01，并保持网格、
% 精确解、随机种子、Fourier 离散方式和平方型稳定化泛函：
%       Omega(x) = ||x||_2^2 + ||Delta x||_2^2.
%
% 本数值验证在模型真解 x_true 处定义有效算子误差水平
%       h = ||(A_h-A)x_true|| / Omega(x_true).

clc;clear;
close all;

%% ========================================================================
%  公用参数设置
% =========================================================================
common.delta_rel = 0.01;              % 固定 1% 数据相对误差
common.model_rel_list = 0:0.02:0.20;  % 0%--20% 的物理参数相对扰动
common.kappa = 1.20;                  % 理论中的 kappa > 1
common.check_tol = 5.0e-11;           % 理论不等式数值检查容差

fprintf('\n============================================================\n');
fprintf('固定数据相对误差 delta = %.4f\n',common.delta_rel);
fprintf('固定 kappa = %.2f\n',common.kappa);
fprintf('============================================================\n');

%% ========================================================================
%  实验 1：二维逆热传导问题
% =========================================================================
heat.N = 256;
heat.L = 0.5;
heat.a = 0.2;
heat.tau = 0.05;
heat.t = 0.075;
heat.seed = 20260913;   
heat.C_R = 1.1;

N = heat.N;
x = linspace(-heat.L,heat.L,N+1);
x(end) = [];
[X,Y] = meshgrid(x,x);

% 精确解：中心热源 + 旋转椭圆环形热源
rxy = sqrt(X.^2+Y.^2);
z1 = 0.5*exp(-(rxy/0.1).^2);
theta = 45*pi/180;
Xr = cos(theta)*X+sin(theta)*Y;
Yr = -sin(theta)*X+cos(theta)*Y;
rho = sqrt((Xr/0.3).^2+(Yr/0.2).^2);
z2 = 0.2*exp(-((rho-1)/0.4).^2);
z_true_heat = z1+z2;
ztrue_hat_heat = fft2(z_true_heat);

% Fourier 波数
k1 = 2*pi*ifftshift((-floor(N/2):ceil(N/2)-1)/(2*heat.L));
[KX,KY] = meshgrid(k1,k1);
k2 = KX.^2+KY.^2;

% 精确热扩散算子与稳定化权重
dt = heat.t-heat.tau;
Ahat_heat = exp(-(heat.a^2)*dt*k2);
Rhat_heat = 1+k2.^2;

% 精确数据
u_true_heat = real(ifft2(Ahat_heat.*ztrue_hat_heat));

% 与 Sec4.1 相同随机种子，固定 1% 数据噪声
rng(heat.seed,'twister');
noise_heat = randn(N,N);
noise_heat = noise_heat/grid_norm(noise_heat);
noise_heat = common.delta_rel*grid_norm(u_true_heat)*noise_heat;
u_delta_heat = u_true_heat+noise_heat;
udelta_hat_heat = fft2(u_delta_heat);
delta_abs_heat = grid_norm(noise_heat);

% 原数值实验中的平方型稳定化泛函 Omega(x)=||x||_2^2+||Delta x||_2^2
Omega_true_heat = spectral_omega(ztrue_hat_heat,Rhat_heat,N);

R_heat = heat.C_R*Omega_true_heat;
kappa_R_heat = common.kappa*R_heat;

n_h = numel(common.model_rel_list);
h_heat = zeros(n_h,1);
hL2_heat = zeros(n_h,1);
operator_action_heat = zeros(n_h,1);
actual_heat = zeros(n_h,1);
psi_true_heat = zeros(n_h,1);
psi_R_heat = zeros(n_h,1);
psi_kR_heat = zeros(n_h,1);

fprintf('\n[实验1] 二维逆热传导问题\n');
fprintf('N=%d, a=%.4f, tau=%.4f, t=%.4f, C_R=%.2f\n', ...
    heat.N,heat.a,heat.tau,heat.t,heat.C_R);
fprintf('Omega(true)=%.6e\n',Omega_true_heat);
fprintf('R=%.6e, kappa*R=%.6e, delta_abs=%.6e\n', ...
    R_heat,kappa_R_heat,delta_abs_heat);
fprintf('\n rel.err(a)          h(paper)         h_L2            operator error        actual residual      Psi_true\n');
fprintf('------------------------------------------------------------------------------------------------------\n');

for j = 1:n_h
    eps_model = common.model_rel_list(j);

    % 用扩散系数误差产生结构化算子扰动
    a_h = heat.a*(1+eps_model);
    Ahat_h = exp(-(a_h^2)*dt*k2);

    % 普通 L2->L2 算子范数，仅作为诊断，不作为图中的 h
    multiplier_diff = abs(Ahat_h-Ahat_heat);
    hL2_heat(j) = max(multiplier_diff(:));

    % 真解处由算子扰动单独造成的误差
    operator_action_heat(j) = spectral_norm((Ahat_h-Ahat_heat).*ztrue_hat_heat,N);

    % 与 ||F_h(x)-F(x)|| <= h*Omega(x) 对齐
    h_heat(j) = operator_action_heat(j)/max(Omega_true_heat,eps);

    % 真解在近似算子和含噪数据下的实际残差
    actual_heat(j) = spectral_norm(Ahat_h.*ztrue_hat_heat-udelta_hat_heat,N);

    % 控制函数：Psi((h,delta),s)=delta_abs+h*s
    psi_true_heat(j) = delta_abs_heat+h_heat(j)*Omega_true_heat;
    psi_R_heat(j) = delta_abs_heat+h_heat(j)*R_heat;
    psi_kR_heat(j) = delta_abs_heat+h_heat(j)*kappa_R_heat;

    % 数值检查
    scale = max([1,psi_kR_heat(j)]);
    if actual_heat(j)>psi_true_heat(j)+common.check_tol*scale
        warning('逆热传导：实际残差超过 Psi(eta,Omega(true))，请检查离散定义。');
    end
    if psi_true_heat(j)>psi_R_heat(j)+common.check_tol*scale || ...
       psi_R_heat(j)>psi_kR_heat(j)+common.check_tol*scale
        warning('逆热传导：Psi 理论链出现数值违背，请检查 R 与 kappa。');
    end

    fprintf('%10.4f     %13.6e   %13.6e   %17.6e   %17.6e   %13.6e\n', ...
        eps_model,h_heat(j),hL2_heat(j),operator_action_heat(j),actual_heat(j),psi_true_heat(j));
end

% 图 1：逆热传导
figure('Color','w','Name','二维逆热传导：算子误差 h 与理论控制量');
plot(h_heat,actual_heat,'-o','LineWidth',1.6,'MarkerSize',6, ...
    'DisplayName','实际残差 ||A_h x_true-y_delta||');
hold on;
plot(h_heat,psi_true_heat,'--s','LineWidth',1.6,'MarkerSize',6, ...
    'DisplayName','Psi(eta,Omega(x_true))');
plot(h_heat,psi_R_heat,'-.^','LineWidth',1.6,'MarkerSize',6, ...
    'DisplayName','Psi(eta,R)');
plot(h_heat,psi_kR_heat,':d','LineWidth',1.8,'MarkerSize',6, ...
    'DisplayName','Psi(eta,kappa R)');
hold off;
grid on;
box on;
xlabel('算子误差水平 h','Interpreter','none');
ylabel('误差/理论容许量','Interpreter','none');
title('二维逆热传导：算子扰动下的理论控制量','Interpreter','none');
legend('Location','northwest','Interpreter','none');
set(gca,'FontSize',11);

%% ========================================================================
%  实验 2：二维位场向下延拓问题
% =========================================================================
pot.zeta = 0.15;
pot.nu = 0.25;
pot.roi_min = 0.0;
pot.roi_max = 1.0;
pot.N_fine = 512;
pot.N = 128;
pot.seed = 1218;
pot.C_R = 1.1;

box_min = pot.roi_min;
box_max = pot.roi_max;
box_length = box_max-box_min;
Nf = pot.N_fine;
N = pot.N;
dxf = box_length/Nf;
xf = box_min+(0:Nf-1)*dxf;
[Xf,Yf] = meshgrid(xf,xf);

% 源函数 w = 20 chi_T1 + chi_T2
T1 = (Xf-0.32).^2+(Yf-0.32).^2 < 0.0004;
T2 = (Xf-0.60).^2-(Xf-0.60).*(Yf-0.60)+(Yf-0.60).^2 < 0.01;
w_fine = 20*double(T1)+double(T2);

% 在 N=512 细网格上构造精确位场，再降采样到 N=128
[~,kabs_fine] = fourier_frequencies_2d(Nf,box_length);
what_fine = fft2(w_fine);
ztrue_fine_hat = exp(-pot.zeta*kabs_fine).*what_fine;
z_true_fine = real(ifft2(ztrue_fine_hat));

if mod(Nf,N)~=0
    error('位场实验要求 N_fine 必须是 N 的整数倍。');
end
stride = Nf/N;
z_true_pot = z_true_fine(1:stride:end,1:stride:end);
ztrue_hat_pot = fft2(z_true_pot);

% 粗网格 Fourier 模型
[k2_pot,kabs_pot] = fourier_frequencies_2d(N,box_length);
d_true = pot.nu-pot.zeta;
Ahat_pot = exp(-d_true*kabs_pot);
Rhat_pot = 1+k2_pot.^2;

% 为严格验证理论不等式，观测数据使用同一粗网格精确算子生成
u_true_pot = real(ifft2(Ahat_pot.*ztrue_hat_pot));

% 细网格产生固定噪声方向，限制到粗网格后标定为 1%
rng(pot.seed,'twister');
noise_direction_fine = randn(Nf,Nf);
noise_direction_fine = noise_direction_fine/grid_norm(noise_direction_fine);
noise_direction_coarse = noise_direction_fine(1:stride:end,1:stride:end);
noise_direction_coarse_norm = grid_norm(noise_direction_coarse);
if noise_direction_coarse_norm<=eps
    error('位场实验：粗网格噪声方向范数过小。');
end
noise_scale_pot = common.delta_rel*grid_norm(u_true_pot)/noise_direction_coarse_norm;
noise_pot = noise_scale_pot*noise_direction_coarse;
u_delta_pot = u_true_pot+noise_pot;
udelta_hat_pot = fft2(u_delta_pot);
delta_abs_pot = grid_norm(noise_pot);

Omega_true_pot = spectral_omega(ztrue_hat_pot,Rhat_pot,N);

R_pot = pot.C_R*Omega_true_pot;
kappa_R_pot = common.kappa*R_pot;

h_pot = zeros(n_h,1);
hL2_pot = zeros(n_h,1);
operator_action_pot = zeros(n_h,1);
actual_pot = zeros(n_h,1);
psi_true_pot = zeros(n_h,1);
psi_R_pot = zeros(n_h,1);
psi_kR_pot = zeros(n_h,1);

fprintf('\n[实验2] 二维位场向下延拓问题\n');
fprintf('N_fine=%d, N=%d, zeta=%.4f, nu=%.4f, d=%.4f, C_R=%.2f\n',pot.N_fine,pot.N,pot.zeta,pot.nu,d_true,pot.C_R);
fprintf('Omega(true)=%.6e\n',Omega_true_pot);
fprintf('R=%.6e, kappa*R=%.6e, delta_abs=%.6e\n',R_pot,kappa_R_pot,delta_abs_pot);
fprintf('\n rel.err(d)          h(paper)         h_L2            operator error        actual residual      Psi_true\n');
fprintf('------------------------------------------------------------------------------------------------------\n');

for j = 1:n_h
    eps_model = common.model_rel_list(j);

    % 用观测/恢复平面间距 d=nu-zeta 的误差产生结构化算子扰动
    d_h = d_true*(1+eps_model);
    Ahat_h = exp(-d_h*kabs_pot);

    multiplier_diff = abs(Ahat_h-Ahat_pot);
    hL2_pot(j) = max(multiplier_diff(:));

    % 真解处由算子扰动单独造成的误差
    operator_action_pot(j) = spectral_norm((Ahat_h-Ahat_pot).*ztrue_hat_pot,N);

    % 与 ||F_h(x)-F(x)|| <= h*Omega(x) 对齐：
    h_pot(j) = operator_action_pot(j)/max(Omega_true_pot,eps);

    actual_pot(j) = spectral_norm(Ahat_h.*ztrue_hat_pot-udelta_hat_pot,N);

    psi_true_pot(j) = delta_abs_pot+h_pot(j)*Omega_true_pot;
    psi_R_pot(j) = delta_abs_pot+h_pot(j)*R_pot;
    psi_kR_pot(j) = delta_abs_pot+h_pot(j)*kappa_R_pot;

    scale = max([1,psi_kR_pot(j)]);
    if actual_pot(j)>psi_true_pot(j)+common.check_tol*scale
        warning('位场延拓：实际残差超过 Psi(eta,Omega(true))，请检查离散定义。');
    end
    if psi_true_pot(j)>psi_R_pot(j)+common.check_tol*scale || psi_R_pot(j)>psi_kR_pot(j)+common.check_tol*scale
        warning('位场延拓：Psi 理论链出现数值违背，请检查 R 与 kappa。');
    end

    fprintf('%10.4f     %13.6e   %13.6e   %17.6e   %17.6e   %13.6e\n', ...
        eps_model,h_pot(j),hL2_pot(j),operator_action_pot(j),actual_pot(j),psi_true_pot(j));
end

% 图 2：位场延拓
figure('Color','w','Name','二维位场延拓：算子误差 h 与理论控制量');
plot(h_pot,actual_pot,'-o','LineWidth',1.6,'MarkerSize',6,'DisplayName','实际残差 ||A_h x_true-y_delta||');
hold on;
plot(h_pot,psi_true_pot,'--s','LineWidth',1.6,'MarkerSize',6,'DisplayName','Psi(eta,Omega(x_true))');
plot(h_pot,psi_R_pot,'-.^','LineWidth',1.6,'MarkerSize',6,'DisplayName','Psi(eta,R)');
plot(h_pot,psi_kR_pot,':d','LineWidth',1.8,'MarkerSize',6,'DisplayName','Psi(eta,kappa R)');
hold off;grid on;box on;
xlabel('算子误差水平 h','Interpreter','none');
ylabel('误差/理论容许量','Interpreter','none');
title('二维位场延拓：算子扰动下的理论控制量','Interpreter','none');
legend('Location','northwest','Interpreter','none');
set(gca,'FontSize',11);

%% ========================================================================
%  汇总说明
% =========================================================================
fprintf('\n============================================================\n');
fprintf('计算完成：共生成两张图。\n');
fprintf('图1：二维逆热传导问题；图2：二维位场向下延拓问题。\n');
fprintf('理论链应满足：\n');
fprintf('  ||A_h x_true-y_delta|| <= Psi(eta,Omega(true))\n');
fprintf('  <= Psi(eta,R) <= Psi(eta,kappa R).\n');
fprintf('============================================================\n');
fprintf('论文对齐定义：||(A_h-A)x|| <= h*Omega(x).\n');
fprintf('控制函数：Psi((h,delta),s)=delta_abs+h*s.\n');

%% ========================================================================
%  局部函数
% =========================================================================
function value = grid_norm(v)
    value = sqrt(mean(abs(v(:)).^2));
end

function value = spectral_norm(vhat,N)
    value = sqrt(real(sum(abs(vhat(:)).^2)))/N^2;
end

function value = spectral_omega(zhat,Rhat,N)
    value = real(sum(Rhat(:).*abs(zhat(:)).^2))/N^4;
end

function [k2,kabs] = fourier_frequencies_2d(N,L)
    if mod(N,2)==0
        modes = [0:N/2-1,-N/2:-1];
    else
        modes = [0:(N-1)/2,-(N-1)/2:-1];
    end
    k1 = (2*pi/L)*modes;
    [KX,KY] = meshgrid(k1,k1);
    k2 = KX.^2+KY.^2;
    kabs = sqrt(k2);
end

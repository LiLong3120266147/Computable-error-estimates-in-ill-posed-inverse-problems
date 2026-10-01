% 4.3: 线性不适定问题后验误差估计的快速算法
%
%   1. 使用二维位场延拓问题、信号和 D1+D2 矩形区域；
%   2. 全局/局部后验估计使用球面梯度快速算法；
%   3. 后验可行集放大系数同时比较四种选取方式：
%      固定单常数 C、可容许单常数 C_min、固定双系数对(C_res,C_Omega)、
%      以及可容许双系数对 (C_res^ad,C_Omega^ad)；
%   4. 自适应系数按“使真解进入后验可行集所需最小系数”计算，并向上保留两位小数；
%   5. 最优恢复误差采用参考代码中的 Bayev-Lagrange Fourier 公式

clear; clc; close all;

%% ============================ 0. 参数设置 ============================

cfg.Nfine = 512;                  % 细网格，用于生成精细周期 Fourier 模型
cfg.N = 128;                      % 反演网格
cfg.zeta = 0.15;                  % 待恢复位场所在平面
cfg.nu = 0.25;                    % 观测数据所在平面

% 第一个代码中的噪声序列
cfg.deltaList = 0.01:0.01:0.10;

cfg.numRealizations = 1;

cfg.randomSeed = 1218;

% 偏差原则常数
cfg.Cdiscrepancy = 1.01;

% 固定 C
cfg.Cfixed.global = 1.23;
cfg.Cfixed.local  = 1.23;

% 固定双系数对
cfg.usePairFixed = true;
cfg.CpairFixed.residual = 1.01;
cfg.CpairFixed.omega = 1.23;

% 自适应 C
cfg.Cadaptive.min = 1.01;
cfg.Cadaptive.max = 5.00;
cfg.Cadaptive.safetySteps = 0;    % 若设为 1，则向上两位小数后再加 0.01

% 是否计算可容许双系数对
cfg.usePairAdaptive = true;

% 是否同时计算固定单常数 C 和可容许单常数 C_min
% 可选：'fixed', 'adaptive', 'both'
cfg.CrunMode = 'both';

% 正则化权重 Omega(z)=<z,Rz>，这里 Rhat=1+(laplacianScale*omega^2)^2
cfg.laplacianScale = 1.0;

% 球面优化参数
cfg.smoothMinPower = 500;
cfg.maxSphereIterations = 200;
cfg.gradientTolerance = 2e-8;
cfg.initialAngleStep = 0.30;
cfg.maxAngleStep = 0.60;
cfg.lineSearchMax = 20;
cfg.randomStarts = 10;
cfg.verboseOptimizer = false;

% 显示 delta 的重建图
cfg.showImagesAtDelta = 0.05;

% 矩形区域
cfg.region1 = [0.20,0.44,0.20,0.44];
cfg.region2 = [0.45,0.75,0.45,0.75];

% 是否复用同一随机方向对应不同 delta
cfg.reuseNoiseDirectionAcrossDeltas = true;

% 是否计算全局后验估计
cfg.computeGlobalPosterior = true;

% 是否计算局部线性泛函的最优恢复误差
cfg.computeOptimalRecovery = true;
cfg.optPriorFactor = 1.00;
cfg.optErrorFactor = 1.00;
cfg.optLogtMin = -40;
cfg.optLogtMax = 40;
cfg.optGridSize = 241;
cfg.optTolX = 1.0e-10;

% 局部后验误差估计方法
% false：默认使用第二个代码的球面梯度快速算法；
% true ：改用 Algorithm 2 线性泛函上下界算法作为对照
cfg.useAlgorithm2ForLocalPosterior = false;
cfg.dualLogtMin = -40;
cfg.dualLogtMax = 12;
cfg.dualGridSize = 161;
cfg.dualTolX = 1.0e-10;
cfg.dualFeasTol = 1.0e-8;

%% ============================ 1. 构造模型 ============================

fprintf('正在构造第一个代码的问题模型：Nfine=%d, N=%d ...\n', cfg.Nfine, cfg.N);
model = buildModel_FirstProblem(cfg);

fprintf('局部区域：D1=[%.2f,%.2f]x[%.2f,%.2f], D2=[%.2f,%.2f]x[%.2f,%.2f]\n', ...
    cfg.region1(1),cfg.region1(2),cfg.region1(3),cfg.region1(4), ...
    cfg.region2(1),cfg.region2(2),cfg.region2(3),cfg.region2(4));

numDelta = numel(cfg.deltaList);

%% ============================ 2. 预分配数组 ============================

trueGlobalAll = zeros(cfg.numRealizations,numDelta);
trueLocalAll  = zeros(cfg.numRealizations,numDelta);
alphaAll      = zeros(cfg.numRealizations,numDelta);
residualAll   = zeros(cfg.numRealizations,numDelta);

postGlobalFixedAll = NaN(cfg.numRealizations,numDelta);
postLocalFixedAll  = NaN(cfg.numRealizations,numDelta);
postGlobalAdaptAll = NaN(cfg.numRealizations,numDelta);
postLocalAdaptAll  = NaN(cfg.numRealizations,numDelta);

postGlobalPairFixedAll = NaN(cfg.numRealizations,numDelta);
postLocalPairFixedAll  = NaN(cfg.numRealizations,numDelta);
postGlobalPairAdaptAll = NaN(cfg.numRealizations,numDelta);
postLocalPairAdaptAll  = NaN(cfg.numRealizations,numDelta);

CfixedGlobalAll = NaN(cfg.numRealizations,numDelta);
CfixedLocalAll  = NaN(cfg.numRealizations,numDelta);
CadaptGlobalAll = NaN(cfg.numRealizations,numDelta);
CadaptLocalAll  = NaN(cfg.numRealizations,numDelta);

CpairFixedResidualAll = NaN(cfg.numRealizations,numDelta);
CpairFixedOmegaAll    = NaN(cfg.numRealizations,numDelta);
CpairAdaptResidualAll = NaN(cfg.numRealizations,numDelta);
CpairAdaptOmegaAll    = NaN(cfg.numRealizations,numDelta);

CneededDataAll = zeros(cfg.numRealizations,numDelta);
CneededRegAll  = zeros(cfg.numRealizations,numDelta);
CneededAll     = zeros(cfg.numRealizations,numDelta);

% 最优恢复误差只用于局部后验估计图中比较
optimalLocalAll = NaN(cfg.numRealizations,numDelta);

% 用 x 标记真解是否落在对应后验可行集合中: 1 表示落在可行集合中，0 表示没有落入
fixedFeasibleAll = NaN(cfg.numRealizations,numDelta);
adaptFeasibleAll = NaN(cfg.numRealizations,numDelta);
pairFixedFeasibleAll = NaN(cfg.numRealizations,numDelta);
pairAdaptFeasibleAll = NaN(cfg.numRealizations,numDelta);

representative = struct();

runFixed = strcmpi(cfg.CrunMode,'fixed') || strcmpi(cfg.CrunMode,'both');
runAdapt = strcmpi(cfg.CrunMode,'adaptive') || strcmpi(cfg.CrunMode,'both');
runPairFixed = isfield(cfg,'usePairFixed') && cfg.usePairFixed;
runPairAdapt = isfield(cfg,'usePairAdaptive') && cfg.usePairAdaptive;

if ~(runFixed || runAdapt || runPairFixed || runPairAdapt)
    error('至少需要开启一种 C 选取方式。');
end

%% ============================ 3. 主循环 ============================

startTime = tic;

for iReal = 1:cfg.numRealizations

    fprintf('\n============================================================\n');
    fprintf('随机噪声实现 %d / %d\n', iReal, cfg.numRealizations);
    fprintf('============================================================\n');

    rng(cfg.randomSeed + iReal - 1, 'twister');

    if cfg.reuseNoiseDirectionAcrossDeltas
        % 先在 Nfine 细网格生成噪声方向，再限制到 N=128 反演网格，并按粗网格噪声范数标定幅值
        commonNoiseFine = randn(cfg.Nfine,cfg.Nfine);
        commonNoiseFine = commonNoiseFine/l2norm(commonNoiseFine);
        commonNoiseCoarse = commonNoiseFine(model.fineToCoarseIndex,model.fineToCoarseIndex);
        commonNoiseCoarseNorm = l2norm(commonNoiseCoarse);
        if commonNoiseCoarseNorm <= eps
            error('限制到粗网格后的随机噪声方向范数过小。');
        end
    end

    previousGlobalFixed = [];
    previousLocalFixed  = [];
    previousGlobalAdapt = [];
    previousLocalAdapt  = [];

    previousGlobalPairFixed = [];
    previousLocalPairFixed  = [];
    previousGlobalPairAdapt = [];
    previousLocalPairAdapt  = [];

    for id = 1:numDelta

        delta = cfg.deltaList(id);

        % 避免跨噪声等级误用上一轮的最优方向变量
        bestG = [];
        bestF = [];
        bestGP = [];
        bestFP = [];
        bestGAP = [];
        bestFAP = [];

        if cfg.reuseNoiseDirectionAcrossDeltas
            noiseScale = delta*l2norm(model.uExact)/commonNoiseCoarseNorm;
            noiseFine = noiseScale*commonNoiseFine;
            uDeltaFine = model.uFine + noiseFine;
            uDelta = uDeltaFine(model.fineToCoarseIndex,model.fineToCoarseIndex);
            noise = uDelta - model.uExact;
        else
            noiseDirection = normalizeL2(randn(cfg.N,cfg.N));
            noise = delta*l2norm(model.uExact)*noiseDirection;
            uDelta = model.uExact + noise;
        end

        % 相对噪声定义: ||noise|| / ||uExact|| = delta
        noiseNorm = l2norm(noise);

        % Tikhonov + 偏差原则
        [zApprox, alpha, residualNorm] = solveTikhonovDiscrepancy(uDelta, model, noiseNorm, cfg);

        alphaAll(iReal,id) = alpha;
        residualAll(iReal,id) = residualNorm;

        % 真实全局相对误差
        trueGlobalAll(iReal,id) = l2norm(zApprox-model.zExact)/max(l2norm(model.zExact),eps);

        % 局部真实相对误差
        exactFunctional = innerL2(model.fWeight,model.zExact);
        trueLocalAll(iReal,id) = abs(innerL2(model.fWeight,zApprox-model.zExact))/ ...
            max(abs(exactFunctional),eps);

        % 自适应 C 所需的诊断量
        %   残差约束： ||Az-u_delta|| <= C ||Az_delta-u_delta||
        %   罚项约束： Omega(z) <= C Omega(z_delta)
        AzExact = applyMultiplier(model.zExact,model.Ahat);
        zApproxHat = fft2(zApprox);
        uDeltaHat = fft2(uDelta);

        RzExact = applyMultiplier(model.zExact,model.Rhat);
        RzApprox = applyMultiplier(zApprox,model.Rhat);

        OmegaExact  = max(innerL2(model.zExact,RzExact),0);
        OmegaApprox = max(innerL2(zApprox,RzApprox),eps);

        trueResidual = l2norm(AzExact-uDelta);

        CneededData = trueResidual/max(residualNorm,eps);
        CneededReg  = OmegaExact/max(OmegaApprox,eps);

        Cneeded = max([1,CneededData,CneededReg]);
        Cadapt = ceil_to_two_decimals(Cneeded);
        Cadapt = min(max(Cadapt,cfg.Cadaptive.min),cfg.Cadaptive.max);
        Cadapt = Cadapt + 0.01*cfg.Cadaptive.safetySteps;

        CpairFixedRes = cfg.CpairFixed.residual;
        CpairFixedOmega = cfg.CpairFixed.omega;

        CadaptRes = ceil_to_two_decimals(max(cfg.Cadaptive.min,CneededData));
        CadaptOmega = ceil_to_two_decimals(max(cfg.Cadaptive.min,CneededReg));
        CadaptRes = min(max(CadaptRes,cfg.Cadaptive.min),cfg.Cadaptive.max);
        CadaptOmega = min(max(CadaptOmega,cfg.Cadaptive.min),cfg.Cadaptive.max);
        CadaptRes = CadaptRes + 0.01*cfg.Cadaptive.safetySteps;
        CadaptOmega = CadaptOmega + 0.01*cfg.Cadaptive.safetySteps;

        CneededDataAll(iReal,id) = CneededData;
        CneededRegAll(iReal,id) = CneededReg;
        CneededAll(iReal,id) = Cneeded;

        % 局部线性泛函的最优恢复误差
        % 使用模型真解先验半径 Omega(z_exact) 和实际信息误差半径 ||Az_exact-u_delta||
        if cfg.computeOptimalRecovery
            trueOmegaForOpt = cfg.optPriorFactor * OmegaExact;
            infoRadiusForOpt = cfg.optErrorFactor * trueResidual;
            optimalOut = optimalRecoveryErrorFourierRef( ...
                model.Ahat,model.Rhat,model.localWeights,trueOmegaForOpt,infoRadiusForOpt,cfg);
            optimalLocalAll(iReal,id) = optimalOut.error/max(abs(exactFunctional),eps);
        end

        % 固定 C
        if runFixed
            Cg = cfg.Cfixed.global;
            Cf = cfg.Cfixed.local;

            CfixedGlobalAll(iReal,id) = Cg;
            CfixedLocalAll(iReal,id) = Cf;

            fixedFeasibleAll(iReal,id) = double(Cf >= Cneeded - 1.0e-12);

            if cfg.computeGlobalPosterior
                posteriorGlobal = buildPosteriorProblem(zApprox,uDelta,model,Cg,Cg,cfg.smoothMinPower);
                startsG = buildStartDirections(zApprox,uDelta,model,cfg,model.emptyWeight);
                if ~isempty(previousGlobalFixed)
                    startsG = [{previousGlobalFixed}, startsG];
                end

                [rhoFixed,bestG] = maximizeOnL2Sphere(posteriorGlobal,startsG,[],cfg);
                previousGlobalFixed = bestG;
                postGlobalFixedAll(iReal,id) = rhoFixed/max(l2norm(model.zExact),eps);
            end

            if cfg.useAlgorithm2ForLocalPosterior
                % 局部后验误差改用 Algorithm 2 直接求后验可行集合上线性泛函的上下界 fmin/fmax
                localFixed = algorithm2RegionFourierFast( ...
                    model.Ahat,model.Rhat,uDeltaHat,zApproxHat, ...
                    Cf*OmegaApprox,Cf*residualNorm,model.regions,cfg.N,cfg);
                postLocalFixedAll(iReal,id) = ...
                    localFixed.E_functional(1)/max(abs(exactFunctional),eps);
                bestF = [];
            else
                posteriorLocal  = buildPosteriorProblem(zApprox,uDelta,model,Cf,Cf,cfg.smoothMinPower);
                startsF = buildStartDirections(zApprox,uDelta,model,cfg,model.fWeight);
                if ~isempty(previousLocalFixed)
                    startsF = [{previousLocalFixed}, startsF];
                end
                [sigmaFixed,bestF] = maximizeOnL2Sphere(posteriorLocal,startsF,model.fWeight,cfg);
                previousLocalFixed = bestF;
                postLocalFixedAll(iReal,id)  = sigmaFixed/max(abs(exactFunctional),eps);
            end
        end

        % 固定双系数对 (C_res,C_Omega)
        if runPairFixed
            CgResP = CpairFixedRes;
            CgOmegaP = CpairFixedOmega;
            CfResP = CpairFixedRes;
            CfOmegaP = CpairFixedOmega;

            CpairFixedResidualAll(iReal,id) = CfResP;
            CpairFixedOmegaAll(iReal,id) = CfOmegaP;

            pairFixedFeasibleAll(iReal,id) = double(CfResP >= CneededData - 1.0e-12 && ...
                CfOmegaP >= CneededReg - 1.0e-12);

            if cfg.computeGlobalPosterior
                posteriorGlobalP = buildPosteriorProblem(zApprox,uDelta,model,CgResP,CgOmegaP,cfg.smoothMinPower);
                startsGP = buildStartDirections(zApprox,uDelta,model,cfg,model.emptyWeight);
                if ~isempty(previousGlobalPairFixed)
                    startsGP = [{previousGlobalPairFixed}, startsGP];
                end
                if runFixed && ~isempty(bestG)
                    startsGP = [{bestG}, startsGP];
                end

                [rhoPairFixed,bestGP] = maximizeOnL2Sphere(posteriorGlobalP,startsGP,[],cfg);
                previousGlobalPairFixed = bestGP;
                postGlobalPairFixedAll(iReal,id) = rhoPairFixed/max(l2norm(model.zExact),eps);
            end

            if cfg.useAlgorithm2ForLocalPosterior
                localPairFixed = algorithm2RegionFourierFast(model.Ahat,model.Rhat,uDeltaHat,zApproxHat, ...
                    CfOmegaP*OmegaApprox,CfResP*residualNorm,model.regions,cfg.N,cfg);
                postLocalPairFixedAll(iReal,id) = localPairFixed.E_functional(1)/max(abs(exactFunctional),eps);
                bestFP = [];
            else
                posteriorLocalP  = buildPosteriorProblem(zApprox,uDelta,model,CfResP,CfOmegaP,cfg.smoothMinPower);
                startsFP = buildStartDirections(zApprox,uDelta,model,cfg,model.fWeight);
                if ~isempty(previousLocalPairFixed)
                    startsFP = [{previousLocalPairFixed}, startsFP];
                end
                if runFixed && ~isempty(bestF)
                    startsFP = [{bestF}, startsFP];
                end

                [sigmaPairFixed,bestFP] = maximizeOnL2Sphere(posteriorLocalP,startsFP,model.fWeight,cfg);
                previousLocalPairFixed = bestFP;
                postLocalPairFixedAll(iReal,id) = sigmaPairFixed/max(abs(exactFunctional),eps);
            end
        end

        % 可容许双系数对 (C_res^ad,C_Omega^ad)
        if runPairAdapt
            CgResAP = CadaptRes;
            CgOmegaAP = CadaptOmega;
            CfResAP = CadaptRes;
            CfOmegaAP = CadaptOmega;

            CpairAdaptResidualAll(iReal,id) = CfResAP;
            CpairAdaptOmegaAll(iReal,id) = CfOmegaAP;

            pairAdaptFeasibleAll(iReal,id) = double(CfResAP >= CneededData - 1.0e-12 && ...
                CfOmegaAP >= CneededReg - 1.0e-12);

            if cfg.computeGlobalPosterior
                posteriorGlobalAP = buildPosteriorProblem(zApprox,uDelta,model,CgResAP,CgOmegaAP,cfg.smoothMinPower);
                startsGAP = buildStartDirections(zApprox,uDelta,model,cfg,model.emptyWeight);
                if ~isempty(previousGlobalPairAdapt)
                    startsGAP = [{previousGlobalPairAdapt}, startsGAP];
                end
                if runPairFixed && ~isempty(bestGP)
                    startsGAP = [{bestGP}, startsGAP];
                elseif runFixed && ~isempty(bestG)
                    startsGAP = [{bestG}, startsGAP];
                end

                [rhoPairAdapt,bestGAP] = maximizeOnL2Sphere(posteriorGlobalAP,startsGAP,[],cfg);
                previousGlobalPairAdapt = bestGAP;
                postGlobalPairAdaptAll(iReal,id) = rhoPairAdapt/max(l2norm(model.zExact),eps);
            end

            if cfg.useAlgorithm2ForLocalPosterior
                localPairAdapt = algorithm2RegionFourierFast(model.Ahat,model.Rhat,uDeltaHat,zApproxHat, ...
                    CfOmegaAP*OmegaApprox,CfResAP*residualNorm,model.regions,cfg.N,cfg);
                postLocalPairAdaptAll(iReal,id) = localPairAdapt.E_functional(1)/max(abs(exactFunctional),eps);
                bestFAP = [];
            else
                posteriorLocalAP  = buildPosteriorProblem(zApprox,uDelta,model,CfResAP,CfOmegaAP,cfg.smoothMinPower);
                startsFAP = buildStartDirections(zApprox,uDelta,model,cfg,model.fWeight);
                if ~isempty(previousLocalPairAdapt)
                    startsFAP = [{previousLocalPairAdapt}, startsFAP];
                end
                if runPairFixed && ~isempty(bestFP)
                    startsFAP = [{bestFP}, startsFAP];
                elseif runFixed && ~isempty(bestF)
                    startsFAP = [{bestF}, startsFAP];
                end

                [sigmaPairAdapt,bestFAP] = maximizeOnL2Sphere(posteriorLocalAP,startsFAP,model.fWeight,cfg);
                previousLocalPairAdapt = bestFAP;
                postLocalPairAdaptAll(iReal,id) = sigmaPairAdapt/max(abs(exactFunctional),eps);
            end
        end

        % 自适应 C
        if runAdapt
            % 全局和局部使用同一个自适应 C, 若需要局部单独设置, 可将 CgA 和 CfA 分开
            CgA = Cadapt;
            CfA = Cadapt;

            CadaptGlobalAll(iReal,id) = CgA;
            CadaptLocalAll(iReal,id)  = CfA;

            adaptFeasibleAll(iReal,id) = double(CfA >= Cneeded - 1.0e-12);

            if cfg.computeGlobalPosterior
                posteriorGlobalA = buildPosteriorProblem(zApprox,uDelta,model,CgA,CgA,cfg.smoothMinPower);
                startsGA = buildStartDirections(zApprox,uDelta,model,cfg,model.emptyWeight);
                if ~isempty(previousGlobalAdapt)
                    startsGA = [{previousGlobalAdapt}, startsGA];
                end
                if runFixed && ~isempty(bestG)
                    startsGA = [{bestG}, startsGA];
                end

                [rhoAdapt,bestGA] = maximizeOnL2Sphere(posteriorGlobalA,startsGA,[],cfg);
                previousGlobalAdapt = bestGA;
                postGlobalAdaptAll(iReal,id) = rhoAdapt/max(l2norm(model.zExact),eps);
            end

            if cfg.useAlgorithm2ForLocalPosterior
                localAdapt = algorithm2RegionFourierFast(model.Ahat,model.Rhat,uDeltaHat,zApproxHat, ...
                    CfA*OmegaApprox,CfA*residualNorm,model.regions,cfg.N,cfg);
                postLocalAdaptAll(iReal,id) = localAdapt.E_functional(1)/max(abs(exactFunctional),eps);
                bestFA = [];
            else
                posteriorLocalA  = buildPosteriorProblem(zApprox,uDelta,model,CfA,CfA,cfg.smoothMinPower);
                startsFA = buildStartDirections(zApprox,uDelta,model,cfg,model.fWeight);
                if ~isempty(previousLocalAdapt)
                    startsFA = [{previousLocalAdapt}, startsFA];
                end
                if runFixed && ~isempty(bestF)
                    startsFA = [{bestF}, startsFA];
                end
                [sigmaAdapt,bestFA] = maximizeOnL2Sphere(posteriorLocalA,startsFA,model.fWeight,cfg);
                previousLocalAdapt = bestFA;
                postLocalAdaptAll(iReal,id)  = sigmaAdapt/max(abs(exactFunctional),eps);
            end
        end

        fprintf(['delta=%5.2f, alpha=%9.3e, res=%8.2e, ','trueG=%7.4f, trueF=%7.4f, Cneed=%.4f, Cadapt=%.2f'], ...
            delta,alpha,residualNorm,trueGlobalAll(iReal,id),trueLocalAll(iReal,id),Cneeded,Cadapt);

        if runFixed
            fprintf(', fixedG=%7.4f, fixedF=%7.4f',postGlobalFixedAll(iReal,id),postLocalFixedAll(iReal,id));
        end
        if runAdapt
            fprintf(', CminG=%7.4f, CminF=%7.4f',postGlobalAdaptAll(iReal,id),postLocalAdaptAll(iReal,id));
        end
        if runPairFixed
            fprintf(', pairFixG=%7.4f, pairFixF=%7.4f',postGlobalPairFixedAll(iReal,id),postLocalPairFixedAll(iReal,id));
        end
        if runPairAdapt
            fprintf(', pairAdG=%7.4f, pairAdF=%7.4f',postGlobalPairAdaptAll(iReal,id),postLocalPairAdaptAll(iReal,id));
        end
        fprintf('\n');

        if iReal==1 && abs(delta-cfg.showImagesAtDelta)<1e-12
            representative.delta = delta;
            representative.uDelta = uDelta;
            representative.zApprox = zApprox;
            representative.error = zApprox-model.zExact;
        end
    end
end

elapsedTime = toc(startTime);

%% ============================ 4. 平均结果 ============================

trueGlobal = mean(trueGlobalAll,1);
trueLocal  = mean(trueLocalAll,1);
alphaMean  = mean(alphaAll,1);

postGlobalFixed = mean(postGlobalFixedAll,1,'omitnan');
postLocalFixed  = mean(postLocalFixedAll,1,'omitnan');
postGlobalAdapt = mean(postGlobalAdaptAll,1,'omitnan');
postLocalAdapt  = mean(postLocalAdaptAll,1,'omitnan');

postGlobalPairFixed = mean(postGlobalPairFixedAll,1,'omitnan');
postLocalPairFixed  = mean(postLocalPairFixedAll,1,'omitnan');
postGlobalPairAdapt = mean(postGlobalPairAdaptAll,1,'omitnan');
postLocalPairAdapt  = mean(postLocalPairAdaptAll,1,'omitnan');

CneededMean = mean(CneededAll,1);
CneededDataMean = mean(CneededDataAll,1);
CneededRegMean = mean(CneededRegAll,1);

CfixedGlobalMean = mean(CfixedGlobalAll,1,'omitnan');
CfixedLocalMean  = mean(CfixedLocalAll,1,'omitnan');
CadaptGlobalMean = mean(CadaptGlobalAll,1,'omitnan');
CadaptLocalMean  = mean(CadaptLocalAll,1,'omitnan');

CpairFixedResidualMean = mean(CpairFixedResidualAll,1,'omitnan');
CpairFixedOmegaMean    = mean(CpairFixedOmegaAll,1,'omitnan');
CpairAdaptResidualMean = mean(CpairAdaptResidualAll,1,'omitnan');
CpairAdaptOmegaMean    = mean(CpairAdaptOmegaAll,1,'omitnan');

optimalLocal = mean(optimalLocalAll,1,'omitnan');
fixedFeasibleRate = mean(fixedFeasibleAll,1,'omitnan');
adaptFeasibleRate = mean(adaptFeasibleAll,1,'omitnan');
pairFixedFeasibleRate = mean(pairFixedFeasibleAll,1,'omitnan');
pairAdaptFeasibleRate = mean(pairAdaptFeasibleAll,1,'omitnan');

resultTable = table(cfg.deltaList(:),alphaMean(:),trueGlobal(:),trueLocal(:),optimalLocal(:), ...
    postGlobalFixed(:),postLocalFixed(:),postGlobalAdapt(:),postLocalAdapt(:), ...
    postGlobalPairFixed(:),postLocalPairFixed(:),postGlobalPairAdapt(:),postLocalPairAdapt(:), ...
    CfixedGlobalMean(:),CfixedLocalMean(:),CadaptGlobalMean(:),CadaptLocalMean(:), ...
    CpairFixedResidualMean(:),CpairFixedOmegaMean(:), ...
    CpairAdaptResidualMean(:),CpairAdaptOmegaMean(:), ...
    CneededDataMean(:),CneededRegMean(:),CneededMean(:), ...
    fixedFeasibleRate(:),adaptFeasibleRate(:),pairFixedFeasibleRate(:),pairAdaptFeasibleRate(:), ...
    'VariableNames', {'delta','alpha','trueGlobal','trueLocal','optimalLocal', ...
    'postGlobalFixed','postLocalFixed','postGlobalCmin','postLocalCmin', ...
    'postGlobalPairFixed','postLocalPairFixed','postGlobalPairAdaptive','postLocalPairAdaptive', ...
    'CfixedGlobal','CfixedLocal','CminGlobal','CminLocal', ...
    'CpairFixedResidual','CpairFixedOmega','CpairAdaptiveResidual','CpairAdaptiveOmega', ...
    'CneededData','CneededRegularizer','Cneeded', ...
    'fixedFeasibleRate','CminFeasibleRate','pairFixedFeasibleRate','pairAdaptiveFeasibleRate'});

fprintf('\n======================== 平均结果 ========================\n');
disp(resultTable);
fprintf('总运行时间：%.2f s\n', elapsedTime);

writetable(resultTable,'Example8_6_using_second_code_results.csv');

save('Example8_6_using_second_code_results.mat', ...
    'cfg','model','resultTable', ...
    'trueGlobalAll','trueLocalAll', ...
    'postGlobalFixedAll','postLocalFixedAll', ...
    'postGlobalAdaptAll','postLocalAdaptAll', ...
    'postGlobalPairFixedAll','postLocalPairFixedAll', ...
    'postGlobalPairAdaptAll','postLocalPairAdaptAll', ...
    'optimalLocalAll','fixedFeasibleAll','adaptFeasibleAll', ...
    'pairFixedFeasibleAll','pairAdaptFeasibleAll', ...
    'CneededAll','CneededDataAll','CneededRegAll', ...
    'CpairFixedResidualAll','CpairFixedOmegaAll', ...
    'CpairAdaptResidualAll','CpairAdaptOmegaAll');

%% ============================ 5. 作图 ============================




% 紫色：可容许 C_min 后验估计
% 青蓝色：C_Omega
% 绿色：C_res
% 黄色 x：真解不在对应后验可行集合中
colorTrue  = [0.0000 0.4470 0.7410];      % 蓝色：近似解真实误差
colorOpt   = [0.2000 0.2000 0.2000];      % 深灰：最优恢复误差
colorFixed = [0.8500 0.3250 0.0980];      % 橙色：固定 C 后验估计
colorAdapt = [0.8000 0.0000 0.0000];      % 紫色：可容许 C_min 后验估计
colorPairFixed = [0.0000 0.6200 0.4510];  
colorPairAdapt = [0.4940 0.1840 0.5560];  
colorOmega = [0.8500 0.3250 0.0980];      % 青蓝色：C_Omega
colorRes   = [0.3000 0.3000 0.3000];      % 绿色：C_res
colorBad   = [0.1000 0.1000 0.1000];      % 黄色 x：真解不在对应后验可行集合中

% -------------------- 图1：全局后验误差估计 --------------------
figure('Name','Global posterior estimate','Color','w');
plot(cfg.deltaList,trueGlobal,'^-','Color',colorTrue,'LineWidth',1.5,'MarkerSize',6, ...
    'DisplayName','近似解全局相对误差');
hold on;

if runFixed
    plot(cfg.deltaList,postGlobalFixed,'o-','Color',colorFixed,'LineWidth',1.5,'MarkerSize',6, ...
        'DisplayName',sprintf('固定 C=%.2f 的全局后验估计',cfg.Cfixed.global));

    badFixed = fixedFeasibleRate < 1;
    if any(badFixed)
        plot(cfg.deltaList(badFixed),postGlobalFixed(badFixed),'x','Color',colorBad, ...
            'LineStyle','none','LineWidth',2.0,'MarkerSize',9,'DisplayName','真解不在固定 C 可行集合中');
    end
end

if runAdapt
    plot(cfg.deltaList,postGlobalAdapt,'s-','Color',colorAdapt,'LineWidth',1.5,'MarkerSize',6, ...
        'DisplayName','可容许 C_{min} 的全局后验估计');

    badAdapt = adaptFeasibleRate < 1;
    if any(badAdapt)
        plot(cfg.deltaList(badAdapt),postGlobalAdapt(badAdapt),'x','Color',colorBad, ...
            'LineStyle','none','LineWidth',2.0,'MarkerSize',9,'DisplayName','真解不在 C_{min} 可行集合中');
    end
end

if runPairFixed
    plot(cfg.deltaList,postGlobalPairFixed,'o:','Color',colorPairFixed,'LineWidth',1.5,'MarkerSize',6, ...
        'DisplayName',sprintf('固定 (C_{res},C_{\\Omega})=(%.2f,%.2f) 的全局后验估计', ...
        cfg.CpairFixed.residual,cfg.CpairFixed.omega));

    badPairFixed = pairFixedFeasibleRate < 1;
    if any(badPairFixed)
        plot(cfg.deltaList(badPairFixed),postGlobalPairFixed(badPairFixed),'x','Color',colorBad, ...
            'LineStyle','none','LineWidth',2.0,'MarkerSize',9,'DisplayName','真解不在固定系数对可行集合中');
    end
end

if runPairAdapt
    plot(cfg.deltaList,postGlobalPairAdapt,'o-','Color',colorPairAdapt,'LineWidth',1.5,'MarkerSize',6, ...
        'DisplayName','可容许 (C_{res},C_{\Omega}) 的全局后验估计');
end

hold off;grid on; box on;
xlabel('相对噪声水平 \delta');
ylabel('相对误差');
title('全局后验误差估计');
legend('Location','northwest','Interpreter','tex');
xlim([cfg.deltaList(1),cfg.deltaList(end)]);
xticks(cfg.deltaList);
set(gca,'FontSize',11);

% -------------------- 图2：局部后验误差估计 --------------------
figure('Name','Local posterior estimate','Color','w');
plot(cfg.deltaList,trueLocal,'^-','Color',colorTrue,'LineWidth',1.5,'MarkerSize',6, ...
    'DisplayName','近似解局部相对误差');
hold on;

if cfg.computeOptimalRecovery
    plot(cfg.deltaList,optimalLocal,'d--','Color',colorOpt,'LineWidth',1.5,'MarkerSize',6, ...
        'DisplayName','最优恢复误差');
end

if runFixed
    plot(cfg.deltaList,postLocalFixed,'o-','Color',colorFixed,'LineWidth',1.5,'MarkerSize',6, ...
        'DisplayName',sprintf('固定 C=%.2f 的局部后验估计',cfg.Cfixed.local));

    badFixed = fixedFeasibleRate < 1;
    if any(badFixed)
        plot(cfg.deltaList(badFixed),postLocalFixed(badFixed),'x','Color',colorBad, ...
            'LineStyle','none','LineWidth',2.0,'MarkerSize',9,'DisplayName','真解不在固定 C 可行集合中');
    end
end

if runAdapt
    plot(cfg.deltaList,postLocalAdapt,'s-','Color',colorAdapt,'LineWidth',1.5,'MarkerSize',6, ...
        'DisplayName','可容许 C_{min} 的局部后验估计');

    badAdapt = adaptFeasibleRate < 1;
    if any(badAdapt)
        plot(cfg.deltaList(badAdapt),postLocalAdapt(badAdapt),'x','Color',colorBad, ...
            'LineStyle','none','LineWidth',2.0,'MarkerSize',9,'DisplayName','真解不在 C_{min} 可行集合中');
    end
end

if runPairFixed
    plot(cfg.deltaList,postLocalPairFixed,'o:','Color',colorPairFixed,'LineWidth',1.5,'MarkerSize',6, ...
        'DisplayName',sprintf('固定 (C_{res},C_{\\Omega})=(%.2f,%.2f) 的局部后验估计', ...
        cfg.CpairFixed.residual,cfg.CpairFixed.omega));

    badPairFixed = pairFixedFeasibleRate < 1;
    if any(badPairFixed)
        plot(cfg.deltaList(badPairFixed),postLocalPairFixed(badPairFixed),'x','Color',colorBad, ...
            'LineStyle','none','LineWidth',2.0,'MarkerSize',9,'DisplayName','真解不在固定系数对可行集合中');
    end
end

if runPairAdapt
    plot(cfg.deltaList,postLocalPairAdapt,'o-','Color',colorPairAdapt,'LineWidth',1.5,'MarkerSize',6, ...
        'DisplayName','可容许 (C_{res},C_{\Omega}) 的局部后验估计');
end

hold off;grid on; box on;
xlabel('相对噪声水平 \delta');
ylabel('相对误差');
title('D_1+D_2 局部后验误差估计');
legend('Location','northwest','Interpreter','tex');
xlim([cfg.deltaList(1),cfg.deltaList(end)]);
xticks(cfg.deltaList);
set(gca,'FontSize',11);

% -------------------- 图3：不同 C 选取方式的系数变化 --------------------
figure('Name','Posterior coefficient C','Color','w');

plot(cfg.deltaList,CneededDataMean,'v-','Color',colorRes,'LineWidth',1.5,'MarkerSize',6, ...
    'DisplayName','C_{res}');
hold on;

plot(cfg.deltaList,CneededRegMean,'o--','Color',colorOmega,'LineWidth',1.5,'MarkerSize',6, ...
    'DisplayName','C_{\Omega}');

plot(cfg.deltaList,CneededMean,'s-','Color',colorTrue,'LineWidth',1.8,'MarkerSize',6, ...
    'DisplayName','C_{min}');

if runFixed
    plot(cfg.deltaList,CfixedLocalMean,':','Color',[0.55 0.55 0.55],'LineWidth',1.5, ...
        'DisplayName',sprintf('固定 C=%.2f',cfg.Cfixed.local));
end

if runPairFixed
    plot(cfg.deltaList,CpairFixedResidualMean,'--','Color',[0.40 0.40 0.40],'LineWidth',1.2, ...
        'DisplayName',sprintf('固定 C_{res}=%.2f',cfg.CpairFixed.residual));
    plot(cfg.deltaList,CpairFixedOmegaMean,'--','Color',colorFixed,'LineWidth',1.2, ...
        'DisplayName',sprintf('固定 C_{\\Omega}=%.2f',cfg.CpairFixed.omega));
end

hold off;grid on; box on;
xlabel('相对噪声水平 \delta');
ylabel('可行集合放大系数 C');
title('不同后验可行集合放大系数的比较');
legend('Location','northwest','Interpreter','tex');
xlim([cfg.deltaList(1),cfg.deltaList(end)]);
xticks(cfg.deltaList);
set(gca,'FontSize',11);

% -------------------- 图4：精确解和矩形区域 --------------------
figure('Name','Exact solution and local rectangles','Color','w');
imagesc(model.x,model.x,model.zExact);
axis image xy; colorbar; hold on;
rectangle('Position',[cfg.region1(1),cfg.region1(3), ...
    cfg.region1(2)-cfg.region1(1),cfg.region1(4)-cfg.region1(3)], ...
    'EdgeColor','w','LineWidth',1.5,'LineStyle','--');
rectangle('Position',[cfg.region2(1),cfg.region2(3), ...
    cfg.region2(2)-cfg.region2(1),cfg.region2(4)-cfg.region2(3)], ...
    'EdgeColor','w','LineWidth',1.5,'LineStyle','--');
hold off;
xlabel('x_1'); ylabel('x_2');
title('Exact field and D_1+D_2 rectangles');

% -------------------- 图5：代表性重建结果 --------------------
if ~isempty(fieldnames(representative))
    figure('Name','Representative reconstruction','Color','w');
    subplot(1,3,1);
    imagesc(model.x,model.x,model.zExact);
    axis image xy; colorbar;
    title('Exact z');

    subplot(1,3,2);
    imagesc(model.x,model.x,representative.zApprox);
    axis image xy; colorbar;
    title(sprintf('Tikhonov, \\delta=%.2f',representative.delta));

    subplot(1,3,3);
    imagesc(model.x,model.x,representative.error);
    axis image xy; colorbar;
    title('Error');
end


%% ============================ 局部函数 ============================

function model = buildModel_FirstProblem(cfg)

    if mod(cfg.Nfine,cfg.N)~=0
        error('cfg.Nfine 必须是 cfg.N 的整数倍。');
    end

    % 细网格信号，保持第一个代码不变
    xFine = (0:cfg.Nfine-1)/cfg.Nfine;
    [Xf,Yf] = meshgrid(xFine,xFine);

    T1fine = (Xf-0.32).^2 + (Yf-0.32).^2 < 0.0004;
    T2fine = (Xf-0.60).^2 - (Xf-0.60).*(Yf-0.60) + (Yf-0.60).^2 < 0.01;

    sourceFine = 20*double(T1fine) + 1*double(T2fine);

    [~,omega2Fine] = frequencyGrid(cfg.Nfine);
    kabsFine = sqrt(omega2Fine);

    % 周期 Fourier Poisson 延拓
    zFineHat = exp(-cfg.zeta*kabsFine).*fft2(sourceFine);
    zFine = real(ifft2(zFineHat));

    uFine = real(ifft2(exp(-(cfg.nu-cfg.zeta)*kabsFine).*zFineHat));

    stride = cfg.Nfine/cfg.N;
    index = 1:stride:cfg.Nfine;

    zExact = zFine(index,index);
    uExact = uFine(index,index);

    % 粗网格信息
    x = (0:cfg.N-1)/cfg.N;
    [X,Y] = meshgrid(x,x);

    T1 = (X-0.32).^2 + (Y-0.32).^2 < 0.0004;
    T2 = (X-0.60).^2 - (X-0.60).*(Y-0.60) + (Y-0.60).^2 < 0.01;
    sourceCoarse = 20*double(T1) + 1*double(T2);

    [~,omega2] = frequencyGrid(cfg.N);
    kabs = sqrt(omega2);

    Ahat = exp(-(cfg.nu-cfg.zeta)*kabs);
    Rhat = 1 + (cfg.laplacianScale*omega2).^2;

    % D1+D2 矩形区域
    b1 = cfg.region1;
    b2 = cfg.region2;

    mask1 = X>=b1(1) & X<=b1(2) & Y>=b1(3) & Y<=b1(4);
    mask2 = X>=b2(1) & X<=b2(2) & Y>=b2(3) & Y<=b2(4);
    mask = mask1 | mask2;

    if ~any(mask(:))
        error('D1+D2 矩形区域为空，请检查 cfg.region1 和 cfg.region2。');
    end

    % innerL2(fWeight,z)=区域平均值
    % localWeights 用于 Algorithm 2 ：sum(localWeights(:).*z(:))
    fWeight = double(mask)/mean(double(mask(:)));
    localWeights = double(mask)/sum(mask(:));

    regions = struct();
    regions(1).name = 'D_1+D_2';
    regions(1).mask1 = mask1;
    regions(1).mask2 = mask2;
    regions(1).mask = mask;
    regions(1).weights = localWeights;
    regions(1).true_value = sum(localWeights(:).*zExact(:));

    model.x = x;
    model.zExact = zExact;
    model.uExact = uExact;
    model.uFine = uFine;
    model.fineToCoarseIndex = index;
    model.sourceCoarse = sourceCoarse;

    model.fWeight = fWeight;
    model.localWeights = localWeights;
    model.regions = regions;
    model.localMask = mask;
    model.localMask1 = mask1;
    model.localMask2 = mask2;

    model.Ahat = Ahat;
    model.A2hat = Ahat.^2;
    model.Rhat = Rhat;
    model.omega2 = omega2;

    model.emptyWeight = [];
end

function [k,omega2] = frequencyGrid(N)
    if mod(N,2)~=0
        error('本程序要求 N 为偶数。');
    end
    modes = [0:N/2-1,-N/2:-1];
    [KX,KY] = meshgrid(modes,modes);
    omega2 = (2*pi)^2*(KX.^2+KY.^2);
    k = sqrt(KX.^2+KY.^2);
end

function y = applyMultiplier(x,multiplier)
    y = real(ifft2(multiplier.*fft2(x)));
end

function [zApprox,alpha,residualNorm] = solveTikhonovDiscrepancy(uDelta,model,noiseNorm,cfg)

    target = cfg.Cdiscrepancy*noiseNorm;
    uHat = fft2(uDelta);

    alphaLeft = 1e-20;
    alphaRight = 1e2;

    residualRight = residualAt(alphaRight,uHat,uDelta,model);
    while residualRight < target
        alphaRight = 10*alphaRight;
        if alphaRight > 1e50
            error('无法括住偏差原则参数 alpha。');
        end
        residualRight = residualAt(alphaRight,uHat,uDelta,model);
    end

    for iter = 1:60
        alphaMid = sqrt(alphaLeft*alphaRight);
        residualMid = residualAt(alphaMid,uHat,uDelta,model);

        if residualMid < target
            alphaLeft = alphaMid;
        else
            alphaRight = alphaMid;
        end
    end

    alpha = sqrt(alphaLeft*alphaRight);
    zApprox = tikhonovAt(alpha,uHat,model);
    residualNorm = l2norm(applyMultiplier(zApprox,model.Ahat)-uDelta);
end

function value = residualAt(alpha,uHat,uDelta,model)
    z = tikhonovAt(alpha,uHat,model);
    value = l2norm(applyMultiplier(z,model.Ahat)-uDelta);
end

function z = tikhonovAt(alpha,uHat,model)
    zHat = conj(model.Ahat).*uHat./(abs(model.Ahat).^2 + alpha*model.Rhat);
    z = real(ifft2(zHat));
end

function posterior = buildPosteriorProblem(zApprox,uDelta,model,Cres,Comega,m)

    v = uDelta - applyMultiplier(zApprox,model.Ahat);

    posterior.Ahat = model.Ahat;
    posterior.A2hat = model.A2hat;
    posterior.Rhat = model.Rhat;
    posterior.v = v;
    posterior.Atv = applyMultiplier(v,conj(model.Ahat));
    posterior.Rz = applyMultiplier(zApprox,model.Rhat);

    % 残差约束：||A(zApprox+t w)-uDelta|| <= Cres ||A zApprox-uDelta||
    % 方向步长中的数据半径为 (Cres^2-1)||residual||^2
    posterior.rA2 = (Cres^2-1)*l2norm(v)^2;

    % 稳定化约束：Omega(zApprox+t w) <= Comega Omega(zApprox)
    % 方向步长中的正则化半径为 (Comega-1)Omega(zApprox)
    posterior.rR2 = (Comega-1)*innerL2(zApprox,posterior.Rz);

    posterior.m = m;
end

function starts = buildStartDirections(zApprox,uDelta,model,cfg,localWeight)

    residual = uDelta - applyMultiplier(zApprox,model.Ahat);
    AtResidual = applyMultiplier(residual,conj(model.Ahat));
    smoothResidual = applyMultiplier(AtResidual,1./sqrt(model.Rhat));

    starts = {zApprox,-zApprox,AtResidual,-AtResidual,smoothResidual,-smoothResidual};

    if ~isempty(localWeight)
        starts = [starts,{localWeight,-localWeight}]; %#ok<AGROW>
    end

    oldState = rng;
    smoothScales = [0.01 0.03 0.06 0.10];

    for j = 1:cfg.randomStarts
        q = randn(size(zApprox));
        filter = exp(-smoothScales(1+mod(j-1,numel(smoothScales)))*sqrt(model.omega2));
        q = applyMultiplier(q,filter);
        starts = [starts,{q,-q}]; %#ok<AGROW>
    end

    rng(oldState);

    keep = true(size(starts));
    for j = 1:numel(starts)
        keep(j) = l2norm(starts{j}) > 1e-14;
    end
    starts = starts(keep);
end

function [bestValue,bestW] = maximizeOnL2Sphere(posterior,starts,localWeight,cfg)

    bestLogValue = -inf;
    bestValue = 0;
    bestW = normalizeL2(starts{1});

    for startIndex = 1:numel(starts)

        w = normalizeL2(starts{startIndex});
        [xi,logXi,gradLogXi] = evaluateSmoothRadius(w,posterior,true);

        if isempty(localWeight)
            logObjective = logXi;
        else
            fw = innerL2(localWeight,w);
            if abs(fw)<1e-14
                w = normalizeL2(w+1e-2*localWeight);
                [xi,logXi,gradLogXi] = evaluateSmoothRadius(w,posterior,true);
                fw = innerL2(localWeight,w);
            end
            logObjective = logXi + log(abs(fw)+realmin);
        end

        angleStep = cfg.initialAngleStep;

        for iter = 1:cfg.maxSphereIterations

            if isempty(localWeight)
                gradient = gradLogXi;
            else
                fw = innerL2(localWeight,w);
                if abs(fw)<1e-14
                    break;
                end
                gradient = gradLogXi + localWeight/fw;
            end

            gradient = gradient - innerL2(w,gradient)*w;
            gradientNorm = l2norm(gradient);

            if gradientNorm < cfg.gradientTolerance
                break;
            end

            direction = gradient/gradientNorm;
            trialAngle = min(angleStep,cfg.maxAngleStep);
            accepted = false;

            for lineIter = 1:cfg.lineSearchMax

                wTrial = cos(trialAngle)*w + sin(trialAngle)*direction;
                [xiTrial,logXiTrial] = evaluateSmoothRadius(wTrial,posterior,false);

                if isempty(localWeight)
                    logObjectiveTrial = logXiTrial;
                else
                    fwTrial = innerL2(localWeight,wTrial);
                    logObjectiveTrial = logXiTrial + log(abs(fwTrial)+realmin);
                end

                if logObjectiveTrial >= logObjective + 1e-4*trialAngle*gradientNorm
                    accepted = true;
                    break;
                end

                trialAngle = 0.5*trialAngle;
            end

            if ~accepted
                break;
            end

            w = wTrial;
            xi = xiTrial;
            logObjective = logObjectiveTrial;

            [xi,logXi,gradLogXi] = evaluateSmoothRadius(w,posterior,true);
            angleStep = min(cfg.maxAngleStep,1.5*trialAngle);

            if trialAngle*gradientNorm < cfg.gradientTolerance
                break;
            end
        end

        if isempty(localWeight)
            candidateValue = xi;
            candidateLogValue = logXi;
        else
            fw = innerL2(localWeight,w);
            candidateValue = xi*abs(fw);
            candidateLogValue = logXi + log(abs(fw)+realmin);
        end

        if cfg.verboseOptimizer
            fprintf('    start %2d: objective=%9.3e\n',startIndex,candidateValue);
        end

        if candidateLogValue > bestLogValue
            bestLogValue = candidateLogValue;
            bestValue = candidateValue;
            bestW = w;
        end
    end
end

function [xi,logXi,gradLogXi,tA,tR] = evaluateSmoothRadius(w,posterior,needGradient)

    W = fft2(w);

    Aw = real(ifft2(posterior.Ahat.*W));
    A2w = real(ifft2(posterior.A2hat.*W));
    Rw = real(ifft2(posterior.Rhat.*W));

    a2 = max(innerL2(Aw,Aw),realmin);
    b = innerL2(Aw,posterior.v);
    sqrtDA = sqrt(max(b^2+a2*posterior.rA2,realmin));
    tA = max((b+sqrtDA)/a2,realmin);

    r2 = max(innerL2(w,Rw),realmin);
    c = innerL2(w,posterior.Rz);
    sqrtDR = sqrt(max(c^2+r2*posterior.rR2,realmin));
    tR = max((-c+sqrtDR)/r2,realmin);

    % 光滑 min(tA,tR)
    logTA = log(tA);
    logTR = log(tR);
    maxLog = max(logTA,logTR);

    exponentA = posterior.m*(logTA-maxLog);
    exponentR = posterior.m*(logTR-maxLog);

    expA = exp(max(exponentA,-745));
    expR = exp(max(exponentR,-745));
    denominator = expA + expR;

    weightA = expA/denominator;
    weightR = expR/denominator;

    logXi = logTA + logTR - maxLog - log(denominator)/posterior.m;
    xi = exp(logXi);

    if ~needGradient
        gradLogXi = [];
        return;
    end

    gradTA = tA*(posterior.Atv - tA*A2w)/sqrtDA;
    gradTR = -tR*(posterior.Rz + tR*Rw)/sqrtDR;

    gradLogXi = weightR*gradTA/tA + weightA*gradTR/tR;
end

function value = innerL2(x,y)
    value = mean(real(x(:).*y(:)));
end

function value = l2norm(x)
    value = sqrt(max(innerL2(x,x),0));
end

function x = normalizeL2(x)
    n = l2norm(x);
    if n <= 1e-15
        error('归一化失败：方向范数过小。');
    end
    x = x/n;
end



%% ========================================================================
%  Algorithm 2：区域平均泛函的 Fourier 上下界算法
% =========================================================================
function local = algorithm2RegionFourierFast(Ahat,Rhat,uhat,zeta_hat,R2,Delta,regions,N,cfg)

    n_region = numel(regions);
    n = N^2;

    bhat = conj(Ahat).*uhat;
    u2 = real(sum(abs(uhat(:)).^2))/n;

    logt_grid = linspace(cfg.dualLogtMin,cfg.dualLogtMax,cfg.dualGridSize);
    ng = numel(logt_grid);

    values_plus = inf(n_region,ng);
    values_minus = inf(n_region,ng);

    chat = cell(n_region,1);
    for ir = 1:n_region
        chat{ir} = fft2(regions(ir).weights);
    end

    for kg = 1:ng
        [center,Mhat,a,valid] = supportRegionComponents(logt_grid(kg),Ahat,Rhat,bhat,u2,R2,Delta,N);

        if ~valid
            continue;
        end

        for ir = 1:n_region
            cMc = real(sum(abs(chat{ir}(:)).^2./Mhat(:)))/n;
            if ~isfinite(cMc) || cMc<=0
                continue;
            end

            q = sqrt(a*cMc);
            center_value = sum(regions(ir).weights(:).*center(:));

            values_plus(ir,kg) = center_value+q;
            values_minus(ir,kg) = -center_value+q;
        end
    end

    fmax = zeros(n_region,1);
    fmin = zeros(n_region,1);
    E_plus = zeros(n_region,1);
    E_minus = zeros(n_region,1);
    E_functional = zeros(n_region,1);

    diagnostics = repmat(emptyLocalDiagnostic(),2*n_region,1);
    max_constraint_violation = 0;
    max_relative_duality_gap = 0;

    zeta = real(ifft2(zeta_hat));

    for ir = 1:n_region
        [upper,logt_upper,diag_upper] = refineSupportRegionValue( ...
            +1,regions(ir),values_plus(ir,:),logt_grid,Ahat,Rhat,bhat,uhat,u2,R2,Delta,zeta,N,cfg);

        [negative_lower,logt_lower,diag_lower] = refineSupportRegionValue( ...
            -1,regions(ir),values_minus(ir,:),logt_grid,Ahat,Rhat,bhat,uhat,u2,R2,Delta,zeta,N,cfg);

        fmax(ir) = upper;
        fmin(ir) = -negative_lower;

        fdelta = sum(regions(ir).weights(:).*zeta(:));
        E_plus(ir) = max(fmax(ir)-fdelta,0);
        E_minus(ir) = max(fdelta-fmin(ir),0);
        E_functional(ir) = max(E_plus(ir),E_minus(ir));

        diag_upper.logt = logt_upper;
        diag_upper.t = exp(logt_upper);
        diag_lower.logt = logt_lower;
        diag_lower.t = exp(logt_lower);
        diag_upper.index = ir;
        diag_lower.index = ir;

        diagnostics(2*ir-1) = diag_upper;
        diagnostics(2*ir) = diag_lower;

        max_constraint_violation = max(max_constraint_violation, ...
            max(diag_upper.constraint_violation,diag_lower.constraint_violation));
        max_relative_duality_gap = max(max_relative_duality_gap, ...
            max(diag_upper.relative_duality_gap,diag_lower.relative_duality_gap));
    end

    local = struct();
    local.fmax = fmax;
    local.fmin = fmin;
    local.E_plus = E_plus;
    local.E_minus = E_minus;
    local.E_functional = E_functional;
    local.max_constraint_violation = max_constraint_violation;
    local.max_relative_duality_gap = max_relative_duality_gap;
    local.diagnostics = diagnostics;
end

function [support_value,best_logt,diagInfo] = refineSupportRegionValue( ...
    sign_value,region,coarse_values,logt_grid,Ahat,Rhat,bhat,uhat,u2,R2,Delta,zeta,N,cfg)

    [coarse_best,idx] = min(coarse_values);
    if ~isfinite(coarse_best)
        error('区域局部后验误差的对偶扫描未得到有限值。');
    end

    il = max(1,idx-1);
    ir = min(numel(logt_grid),idx+1);
    left = logt_grid(il);
    right = logt_grid(ir);

    objective = @(p) supportRegionObjective(p,sign_value,region,Ahat,Rhat,bhat,u2,R2,Delta,N);

    if left==right
        best_logt = logt_grid(idx);
        support_value = coarse_best;
    else
        [best_logt,support_value] = fminbnd(objective,left,right, ...
            optimset('Display','off','TolX',cfg.dualTolX,'MaxIter',250));

        if coarse_best<support_value
            best_logt = logt_grid(idx);
            support_value = coarse_best;
        end
    end

    candidate = recoverSupportRegionCandidate( ...
        best_logt,sign_value,region,Ahat,Rhat,bhat,uhat,u2,R2,Delta,N);

    fdelta = sum(region.weights(:).*zeta(:));
    feasible_lower = sign_value*fdelta;

    if candidate.constraint_violation<=cfg.dualFeasTol
        primal_lower = max(feasible_lower,candidate.primal_value);
    else
        primal_lower = feasible_lower;
    end

    relative_gap = max(support_value-primal_lower,0)/max([1,abs(support_value),abs(primal_lower)]);

    diagInfo = emptyLocalDiagnostic();
    diagInfo.sign = sign_value;
    diagInfo.dual_value = support_value;
    diagInfo.primal_value = candidate.primal_value;
    diagInfo.primal_lower_bound = primal_lower;
    diagInfo.relative_duality_gap = relative_gap;
    diagInfo.omega_ratio = candidate.omega/R2;
    diagInfo.residual_ratio = candidate.residual/Delta;
    diagInfo.constraint_violation = candidate.constraint_violation;
end

function [center,Mhat,a,valid] = supportRegionComponents(logt,Ahat,Rhat,bhat,u2,R2,Delta,N)

    center = [];
    Mhat = [];
    a = NaN;
    valid = false;

    if ~isfinite(logt)
        return;
    end

    t = exp(logt);
    Mhat = abs(Ahat).^2+t*Rhat;

    if any(~isfinite(Mhat(:))) || any(Mhat(:)<=0)
        return;
    end

    n = N^2;
    bMb = real(sum(abs(bhat(:)).^2./Mhat(:)))/n;
    a = bMb-u2+n*Delta^2+t*n*R2;

    if ~isfinite(a) || a<=0
        return;
    end

    center = real(ifft2(bhat./Mhat));
    valid = true;
end

function value = supportRegionObjective(logt,sign_value,region,Ahat,Rhat,bhat,u2,R2,Delta,N)

    [center,Mhat,a,valid] = supportRegionComponents(logt,Ahat,Rhat,bhat,u2,R2,Delta,N);

    if ~valid
        value = Inf;
        return;
    end

    n = N^2;
    chat = fft2(region.weights);
    cMc = real(sum(abs(chat(:)).^2./Mhat(:)))/n;

    if ~isfinite(cMc) || cMc<=0
        value = Inf;
        return;
    end

    q = sqrt(a*cMc);
    center_value = sum(region.weights(:).*center(:));
    value = sign_value*center_value+q;
end

function candidate = recoverSupportRegionCandidate( ...
    logt,sign_value,region,Ahat,Rhat,bhat,uhat,u2,R2,Delta,N)

    [~,Mhat,a,valid] = supportRegionComponents(logt,Ahat,Rhat,bhat,u2,R2,Delta,N);

    if ~valid
        error('恢复区域后验极值候选解时出现无效对偶参数。');
    end

    n = N^2;
    chat = sign_value*fft2(region.weights);
    cMc = real(sum(abs(chat(:)).^2./Mhat(:)))/n;

    if cMc<=0
        error('恢复区域后验极值候选解时出现非正二次型。');
    end

    scale = sqrt(a/cMc);
    zhat = (bhat+scale*chat)./Mhat;

    z_candidate = real(ifft2(zhat));
    primal_value = sign_value*sum(region.weights(:).*z_candidate(:));

    omega = spectralOmega(zhat,Rhat,N);
    residual = spectralNorm(Ahat.*zhat-uhat,N);

    violation_omega = max(omega/R2-1,0);
    violation_residual = max(residual/Delta-1,0);

    candidate = struct();
    candidate.primal_value = primal_value;
    candidate.omega = omega;
    candidate.residual = residual;
    candidate.constraint_violation = max(violation_omega,violation_residual);
end

%% ========================================================================
%  Bayev 最优恢复误差
% =========================================================================
function out = optimalRecoveryErrorFourierRef(Ahat,Rhat,weights,Rprior,DeltaInfo,cfg)

    N = size(Ahat,1);
    chat = fft2(weights);
    logt_grid = linspace(cfg.optLogtMin,cfg.optLogtMax,cfg.optGridSize);

    values = zeros(size(logt_grid));
    for k = 1:numel(logt_grid)
        values(k) = optimalRecoveryObjectiveRef(logt_grid(k),Ahat,Rhat,chat,Rprior,DeltaInfo);
    end

    [coarse_best,idx] = min(values);

    il = max(1,idx-1);
    ir = min(numel(logt_grid),idx+1);
    left = logt_grid(il);
    right = logt_grid(ir);

    objective = @(p) optimalRecoveryObjectiveRef(p,Ahat,Rhat,chat,Rprior,DeltaInfo);

    if left==right
        best_logt = logt_grid(idx);
        best_value = coarse_best;
    else
        [best_logt,best_value] = fminbnd(objective,left,right, ...
            optimset('Display','off','TolX',cfg.optTolX,'MaxIter',300));

        if coarse_best<best_value
            best_logt = logt_grid(idx);
            best_value = coarse_best;
        end
    end

    den0 = abs(Ahat).^2;
    value_t0 = DeltaInfo*sqrt(real(sum(abs(chat(:)).^2./den0(:))));

    value_tinf = sqrt(Rprior*real(sum(abs(chat(:)).^2./Rhat(:))));

    [best_value,which_case] = min([best_value,value_t0,value_tinf]);

    if which_case==1
        t_ratio = exp(best_logt);
    elseif which_case==2
        t_ratio = 0;
        best_logt = -Inf;
    else
        t_ratio = Inf;
        best_logt = Inf;
    end

    out = struct();
    out.error = best_value;
    out.t_ratio = t_ratio;
    out.logt = best_logt;
    out.R_prior = Rprior;
    out.Delta_info = DeltaInfo;
    out.case_id = which_case;
end

function value = optimalRecoveryObjectiveRef(logt,Ahat,Rhat,chat,Rprior,DeltaInfo)

    if ~isfinite(logt)
        value = Inf;
        return;
    end

    t = exp(logt);
    den = abs(Ahat).^2+t*Rhat;

    if any(~isfinite(den(:))) || any(den(:)<=0)
        value = Inf;
        return;
    end

    S = real(sum(abs(chat(:)).^2./den(:)));
    value = sqrt((DeltaInfo^2+t*Rprior)*S);
end

function value = spectralNorm(vhat,N)
    value = sqrt(real(sum(abs(vhat(:)).^2)))/N^2;
end

function value = spectralOmega(zhat,Rhat,N)
    value = real(sum(Rhat(:).*abs(zhat(:)).^2))/N^4;
end

function d = emptyLocalDiagnostic()
    d = struct( ...
        'sign',0, ...
        'index',0, ...
        'logt',NaN, ...
        't',NaN, ...
        'dual_value',NaN, ...
        'primal_value',NaN, ...
        'primal_lower_bound',NaN, ...
        'relative_duality_gap',Inf, ...
        'omega_ratio',Inf, ...
        'residual_ratio',Inf, ...
        'constraint_violation',Inf);
end

function value = optimalRecoveryErrorFourier(Ahat,Rhat,fWeight,Rprior,DeltaInfo,cfg)
    % 局部线性泛函 ell(z)=innerL2(fWeight,z) 的最优恢复误差

    N = size(Ahat,1);
    fhat = fft2(fWeight);

    logtGrid = linspace(cfg.optLogtMin,cfg.optLogtMax,cfg.optGridSize);
    values = zeros(size(logtGrid));

    for k = 1:numel(logtGrid)
        values(k) = optimalRecoveryObjective(logtGrid(k),Ahat,Rhat,fhat,Rprior,DeltaInfo,N);
    end

    [coarseBest,idx] = min(values);
    leftIndex = max(1,idx-1);
    rightIndex = min(numel(logtGrid),idx+1);
    left = logtGrid(leftIndex);
    right = logtGrid(rightIndex);

    objective = @(p) optimalRecoveryObjective(p,Ahat,Rhat,fhat,Rprior,DeltaInfo,N);

    if left == right
        bestValue = coarseBest;
    else
        [~,bestValue] = fminbnd(objective,left,right, ...
            optimset('Display','off','TolX',cfg.optTolX,'MaxIter',250));
        bestValue = min(bestValue,coarseBest);
    end

    % 两个边界情形：t=0 和 t -> Inf。
    den0 = abs(Ahat).^2;
    S0 = real(sum(abs(fhat(:)).^2 ./ den0(:))) / N^4;
    valueT0 = DeltaInfo * sqrt(max(S0,0));

    Sinf = real(sum(abs(fhat(:)).^2 ./ Rhat(:))) / N^4;
    valueTinf = sqrt(max(Rprior*Sinf,0));

    value = min([bestValue,valueT0,valueTinf]);
end

function value = optimalRecoveryObjective(logt,Ahat,Rhat,fhat,Rprior,DeltaInfo,N)
    if ~isfinite(logt)
        value = Inf;
        return;
    end

    t = exp(logt);
    den = abs(Ahat).^2 + t*Rhat;

    if any(~isfinite(den(:))) || any(den(:)<=0)
        value = Inf;
        return;
    end

    S = real(sum(abs(fhat(:)).^2 ./ den(:))) / N^4;
    value = sqrt(max((DeltaInfo^2 + t*Rprior)*S,0));
end

function value = ceil_to_two_decimals(x)
    scaled = 100*x;
    nearestInteger = round(scaled);
    tol = 1e-10*max(1,abs(scaled));

    if abs(scaled-nearestInteger)<=tol
        value = nearestInteger/100;
    else
        value = ceil(scaled)/100;
    end
end

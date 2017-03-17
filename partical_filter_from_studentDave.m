%Student Dave's particle filter tutorial

%In this tutorial, The Frequentisian Ninja Clan has just run into the mysterious Quail.
%The Quail was just doing it's typical magical quail stuff throughout the forest like, I
%dunno, freeing catapillars from evil monkey spells.
%The ninja clan, knowing the impossible agility of the Quail, began to
%stretch out and prepare for attack.  During which the Quail mediated and
%generated a highly non-linear flight model with strong illusions (i.e. weird measurements :P) and takes off!
%
%The student ninja's make chase but, only knowing linear algorithms, they
%fail very quickly.  However, the Master Frequentisian Ninja knows the
%particle filter, and after a cognitively and physical exhaustive, epic
%chase, the Master catches the Quail, and takes it back to their secret
%Dojo.

%Here, we learn this master skill, known as the particle filter, as applied
%to a highly nonlinear model. :)!

%Adapted from Dan Simon Optimal state estimation book and Gordon, Salmond,
%and Smith paper.

%% clear everything
clear all
close all
clc


%% initialize the variables
set(0,'DefaultFigureWindowStyle','docked') %dock the figures..just a personal preference you don't need this.
x = 0.1; % initial actual state
x_N = 1; % Noise covariance in the system (i.e. process noise in the state update, here, we'll use a gaussian.)
x_R = 1; % Noise covariance in the measurement (i.e. the Quail creates complex illusions in its trail!)
T = 100; % duration the chase (i.e. number of iterations).
N = 10; % The number of particles the system generates. The larger this is, the better your approximation, but the more computation you need.
D = 2;

%initilize our initial, prior particle distribution as a gaussian around
%the true initial value

Rs = zeros(2,T);
Rs(:,1) = [50; 30];
rs = zeros(2,T);
rs(:,1) = Rs(:,1) + sqrt(x_R) * randn(2,1);
for m = 2:T
    Rs(:,m) = 0.5 * Rs(:,m-1) + 25 * Rs(:,m-1) ./ (1 + Rs(:,m-1).^2) + 8 * cos(1.2 * (m - 1)) + sqrt(x_N) * randn(2,1);
    rs(:,m) = Rs(:,m) + sqrt(x_R) * randn(2,1); 
end

V = 2; %define the variance of the initial esimate
x_P = []; % define the vector of particles

% make the randomly generated particles from the initial prior gaussian distribution
for i = 1:N
    x_P(:,i) = Rs(:,1) + sqrt(V) * randn(2,1);
    P_w(:,i) = [1/N; 1/N];
end

%the functions used by the Quail are:
% x = 0.5*x + 25*x/(1 + x^2) + 8*cos(1.2*(t-1)) + PROCESS NOISE --> sqrt(x_N)*randn
% z = x^2/20 + MEASUREMENT NOISE -->  sqrt(x_R)*randn;

%generate the observations from the randomly selected particles, based upon
%the given function
z_out = zeros(2,T);  %the actual output vector for measurement values.
x_out = zeros(2,T);  %the actual output vector for measurement values.
x_est = zeros(2,1); % time by time output of the particle filters estimate
x_est_out = zeros(2,T); % the vector of particle filter estimates.

for t = 1:T
    %from the previou time step, update the flight position, and observed
    %position (i.e. update the Quails position with the non linear function
    %and update from this position what the chasing ninja's see confounded
    %by the Quails illusions.
%     x = 0.5*x + 25*x/(1 + x^2) + 8*cos(1.2*(t-1)) +  sqrt(x_N)*randn;
%     z = x + sqrt(x_R)*randn;
    x = Rs(:,t) + sqrt(x_N)*randn(2,1);
    z = rs(:,t);
    %Here, we do the particle filter
    for i = 1:N
        %given the prior set of particle (i.e. randomly generated locations
        %the quail might be), run each of these particles through the state
        %update model to make a new set of transitioned particles.
        x_P_update(:,i) = 0.5*x_P(:,i) + 25*x_P(:,i)./(1 + x_P(:,i).^2) + 8*cos(1.2*(t-1)) + sqrt(x_N)*randn(2,1);
        %with these new updated particle locations, update the observations
        %for each of these particles.
%         z_update(i) = x_P_update(i)^2/20;
        z_update(:,i) = x_P_update(:,i);  
        %Generate the weights for each of these particles.
        %The weights are based upon the probability of the given
        %observation for a particle, GIVEN the actual observation.
        %That is, if we observe a location z, and we know our observation error is
        %guassian with variance x_R, then the probability of seeing a given
        %z centered at that actual measurement is (from the equation of a
        %gaussian)
        % 这里，你可以看一下：i为粒子数，z_update(:,i)是第i个粒子更新的观测量，z是真实观测值，P_w(:,i)是第i个粒子的权重
%         i
%         z_update(:,i)
%         z
%         P_w(:,i) = P_w(:,i) .* (1/sqrt(2*pi*x_R)) .* exp(-(z - z_update(:,i)).^2./(2*x_R));
        P_w(:,i) = (1/sqrt(2*pi*x_R)) .* exp(-(z - z_update(:,i)).^2./(2*x_R));
%         P_w(:,i)
    end
    
    % Normalize to form a probability distribution (i.e. sum to 1).
    for a = 1:D
        P_w(a,:) = P_w(a,:) ./ sum(P_w(a,:));
    end
    
    %% Resampling: From this new distribution, now we randomly sample from it to generate our new estimate particles
    
    %what this code specifically does is randomly, uniformally, sample from
    %the cummulative distribution of the probability distribution
    %generated by the weighted vector P_w.  If you sample randomly over
    %this distribution, you will select values based upon there statistical
    %probability, and thus, on average, pick values with the higher weights
    %(i.e. high probability of being correct given the observation z).
    %store this new value to the new estimate which will go back into the
    %next iteration
    for a = 1:D
        cum = cumsum(P_w(a,:));
        for i = 1:N
            x_P(a,i) = x_P_update(a, find(rand <= cum, 1));
        end
    end
    % 这里你可以看一下怎么重采样粒子，涉及到一个累积概率密度：x_P_update是未采样前的状态
    % P_w是权重，cum是累积概率密度，x_P是重采样之后的粒子
%     x_P_update
%     P_w
%     cum
%     x_P
    
    %The final estimate is some metric of these final resampling, such as
    %the mean value or variance
    x_est = mean(x_P, 2);
    
    % Save data in arrays for later plotting
    x_out(:,t) = x;
    z_out(:,t) = z;
    x_est_out(:,t) = x_est;
    
end

t = 1:T;
figure(1); clf
subplot(211);
plot(t, x_out(1,:), '.-b', t, z_out(1,:), '*b', t, x_est_out(1,:), '-.r','linewidth',1);
subplot(212);
plot(t, x_out(2,:), '.-b', t, z_out(2,:), '*b', t, x_est_out(2,:), '-.r','linewidth',1);
set(gca,'FontSize',12); set(gcf,'Color','White');
xlabel('time step'); ylabel('Quail flight position');
legend('True flight position', 'measure position', 'Particle filter estimate');
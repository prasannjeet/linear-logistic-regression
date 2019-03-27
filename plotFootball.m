function plotFootball(X, y, varargin)

pos = find(y==1);
neg = find(y==0);

clf
rgb = imread('soccer.jpg');
imshow(rgb)

hold on;
% plotting Swedish receivers
plot(X(pos,1), X(pos,2), 'ko','MarkerFaceColor','y')
%plotting opposite team receivers
plot(X(neg,1), X(neg,2), 'ko', 'MarkerFaceColor','r')

if nargin==5
    beta = varargin{1};
    mu = varargin{2};
    sigma = varargin{3};
    
    % end points to compute the desicion boundary on normalized data
    plot_x = [(0-mu(1))/sigma(1), (size(rgb,2)-mu(1))/sigma(1)];
    plot_y = (-1./beta(3)).*(beta(2).*(plot_x)+ beta(1));
    
    % renormalization
    plot_x = sigma(1)*plot_x + mu(1);
    plot_y = sigma(2)*plot_y+mu(2);

    plot(plot_x, plot_y, 'g', 'LineWidth', 4)
    legend('Swedish player receiving ball', 'Opposite team receiving ball', 'Decision boundary')
else
    legend('Swedish player receiving ball', 'Opposite team receiving ball')
end
hold off;

end



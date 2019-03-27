classdef ExTwoFunctions
    
    properties
    end
    
    methods(Static)
        
        function clear()
            clear;
            close all;
            clc;
        end
        
        function [beta, costArray, i] = gradientDescent (features, y, a)
        % *Gradient Descent* Function 
        % This function returns the $\beta$ vector according to the features and
        % solution matrix given. It uses gradient descent to calculate $\beta$.
        % This is a generic function and can be used for any number of features.
        %
        % Inputs:
        % 
        % # features: The feature matrix where each row contains one sample entry
        % and each column contains unique features. Note: *ones* vector should not
        % be concatenated, as it will be done here.
        % # y: The solution vector, where each row contains the solution for the
        % corresponding sample entry.

        %% 
        % Converting the feature matrix to 'X' by adding *ones* vector

            [rows, columns] = size(features);
            X = [ones(rows,1) features];

        %% Initializing $\alpha$, denoted by a, and $\beta$ denoted by b
        % Change the value below to perform gradient descent with different values
        % of $\alpha$ and $\beta$.

        %   a = 0.000001;
            b = 0;

        %%
        % $\beta$ value using normal equation and the resultant cost:

        bNormal = ExTwoFunctions.normalEquation(features,y);
        costNormal = ExTwoFunctions.mseMultiFeature(features,y,bNormal);

        %%
        % Performing Gradient Descent

            bOld = ones((columns+1), 1)*b; % The assumend beta value
            prevCost = ExTwoFunctions.mseMultiFeature(features,y,bOld); % Will contain previous value of %\beta%
            newCost = prevCost; % Will contain the current cost value in the loop
            bNew = bOld; % Will contain the current beta value in the loop
            i = 1; % Iterator
            costArray(1,:) = [1 prevCost]; % Returned, can be used to plot changes in the cost with iterations
            loopCondition = true;

            while loopCondition
                i = i+1;
                prevCost = newCost;
                bOld = bNew;
                bNew = bOld - a * X' * (X * bOld - y);
                newCost = ExTwoFunctions.mseMultiFeature(features,y,bNew);
                costArray(i,:) = [i newCost];

                %Breaking condition, when the current cost is within $1%$ of final
                %cost of the normal equation
                if abs(newCost - costNormal) < (1/100)*costNormal
                    loopCondition = false;
                end

                % Following commented lines can be used for debugging, to know how
                % many iterations have occured and how sharply is the cost
                % decreasing.

                % if rem(i,100000) == 0
                %     i
                %     newCost
                % end
            end
            beta = bNew;
		end
        
        function [beta] = mseMultiFeature(features, y, b)
        % *Mean Squared Error* Function
        % This function calculates the Mean Squared Error for any type of datasets,
        % for linear regression.
        %
        % Note that the mean squared error formula used is the following:
        %
        % $\left( \frac{1}{m}\right) \sum_{i=1}^{n} ( h_{\beta}(x) - y _{i})^{2}$
        % 
        % Where, for 'n' features, $h_{\beta}(x)$ can be:
        %
        % $\beta _{0} + \beta _{1} x _{1} + \beta _{2} x _{2} + ... + \beta _{n} x
        % _{n}$
        %
        % Inputs:
        % 
        % # X: A matrix with each test sample in one row and each feature in one
        % column
        % # y: Solution for each test sample.
        % # b: The value of $\beta$ 
        % 
        % Output: The mean squared error

        % Adding one extra column of ones in the feature matrix
        [rows, ~] = size(features);
        X = [ones(rows,1) features];

        % Calculating the MSE
        beta = sum(((sum(((b' * X')'),2)) - y) .^ 2)/rows;
        end
        
        function [beta] = normalEquation(features, y)
        % *NormalEquation* Function
        % This function returns the $\beta$ vector according to the features and solution
        % matrix given. This is a generic function and can be used for any number
        % of features.
        %
        % Inputs:
        % 
        % # features: The feature matrix where each row contains one sample entry
        % and each column contains unique features. Note: *ones* vector should not
        % be concatenated, as it will be done here.
        % # y: The solution vector, where each row contains the solution for the
        % corresponding sample entry.

        % Converting the feature matrix to 'X' by adding *ones* vector
        [rows, ~] = size(features);
        X = [ones(rows,1) features];

        % Now implementing the normal equation
        beta = pinv(X' * X) * X' * y;

        end
        
        function [data] = normalizeData(features)
        % *Normalize Data* Function 
        % This function normalizes the data given. It uses the following formula to
        % normalize: 
        %
        % $\left( \frac{X - \mu}{\sigma}\right)$
        %
        % Where:
        %
        % * $\mu = mean,\quad and$
        % * $\sigma = standard \quad deviation$
        %
        % Steps:
        %
        % # Calculating mean across each column
        % # Repeating the mean vector for each rows in features matrix
        % # Calculating the standard devication across each column
        % # Repeating the standard deviation vector for each row in feature
        % matrix
        % # Substracting the features matrix with mean matrix and then dividing
        % corresponding elements with the standard deviation matrix

            [rows,~] = size(features);
            if rows == 1
                data = features;
                return
            end
            data = (features - repmat(mean(features), rows, 1)) ./ repmat(std(features), rows, 1);
        end
		
		function [cost] = polyCost (features, y, b, d)
		% *Mean Squared Error Calculator for Polynomial Regression*
		% Inputs:
		%
		% # features: the single feature vector (horizontal vector)
		% # y: the solutoin vector
		% # b: the $\beta$ value
		% # d: the maximum degree of polynomial

			x = features';
			[~,samples] = size(x);
			yM = repmat(x,d+1,1);
			for k = 0:d
				yM(k+1,:) = yM(k+1,:) .^ k;
			end
			beetah = repmat(b', 1, samples);
			yM = yM .* beetah;
			yM = sum(yM);
			y = y';
			yM = yM - y;
			yM = yM .^ 2;
			yM = mean(yM);
			cost = yM;
		end
		
		function [cost] = costLogistic (X, y, b)
		% *COST FUNCTION* For logistic regression
		% Inputs:
		%
		% # X: Features Matrix including the ones vector concatenated in the first
		% column
		% # y: The solution vector
		% # b: The value of beta

			[n, ~] = size(X);
			mulVal = X * b;
			cost = (-1/n)*(y'*log(ExTwoFunctions.sigmoid(mulVal))+(1-y)'*log(1-ExTwoFunctions.sigmoid(mulVal)));
		end
		
		function [b,costArray, i] = logisticGradient (X,y,a)
		% *GRADIENT DESCENT FOR LOGISTIC REGRESSION*
		%
		% Inputs:
		%
		% # X: The features array combined with ones vector in the first column.
		% # y: The solutions label.

			% Feature Count will have the total number of features + 1, as it has a
			% ones vector concatenated in the first column
			[n, featureCount] = size(X);
			
			% Initializing variables for the loop
			bNew = zeros(featureCount, 1);
			loopCondition = true;
			i = 1;
			newCost = ExTwoFunctions.costLogistic(X,y,bNew);
			costArray(1,:) = [1, newCost];
			while loopCondition
				i = i+1;
				bOld = bNew;
				prevCost = newCost;
				bNew = bOld - (a/n)*(X')*((ExTwoFunctions.sigmoid(X*bOld))-y);
				newCost = ExTwoFunctions.costLogistic(X,y,bNew);
				
				costArray(i,:) = [i, newCost];
				if (prevCost - newCost) < 0.0000005
					loopCondition = false;
				end
		%         a = a+a/9000000;
		%          if rem(i,1000) == 0                 
		%              i
		%              newCost
		%          end
		%         
		%          if i == 500000
		%              loopCondition = false;
		%          end
			 end
			
			b = bNew;
		end
		function g = sigmoid(z)
			g = 1./(1+exp(-z));
		end
		
		function [jVal, gradient] = costFunctionFminunc (b, football_X, y)
		% *Cost function for fminunc()*

			[n, ~] = size(football_X);
			X = [ones(n,1) football_X];
			mulVal = X * b;
			jVal = (-1/n)*(y'*log(ExTwoFunctions.sigmoid(mulVal))+(1-y)'*log(1-ExTwoFunctions.sigmoid(mulVal)));
			gradient = (1/n)*(X')*(ExTwoFunctions.sigmoid(X*b)-y);
		end
		
		function out = mapFeature(X1, X2, D)
			out = ones(size(X1(:,1)));
			for i = 1:D
				for j = 0:i
					out(:, end+1) = (X1.^(i-j)).*(X2.^j);
				end
			end
		end
		
		function [jVal, gradient] = costFunctionFminuncReg (b, X, y, lambda)
		% *Cost function for fminunc()*
		% Inputs:
		% # b: The beta value
		% # X: The features matrix without ones vector
		% # y: The solution vector
		% # lambda: The regularization constant
		%
		% Output:
		%
		% # jVal: The cost
		% # gradient: The gradient

			[n, ~] = size(X);
			X = [ones(n,1) X];
			mulVal = X * b;
			beta_reg = b(2:end);
			jVal = ((-1/n)*(y'*log(ExTwoFunctions.sigmoid(mulVal))+(1-y)'*log(1-ExTwoFunctions.sigmoid(mulVal)))) + lambda/(2*n)*[beta_reg'*beta_reg];
			gradient = ((1/n)*(X')*(ExTwoFunctions.sigmoid(X*b)-y)) + lambda/n*[0; beta_reg];
		end
		
		function images = loadMNISTImages(filename)
		%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
		%the raw MNIST images

			fp = fopen(filename, 'rb');
			assert(fp ~= -1, ['Could not open ', filename, '']);

			magic = fread(fp, 1, 'int32', 0, 'ieee-be');
			assert(magic == 2051, ['Bad magic number in ', filename, '']);

			numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
			numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
			numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

			images = fread(fp, inf, 'unsigned char');
			images = reshape(images, numCols, numRows, numImages);
			images = permute(images,[2 1 3]);

			fclose(fp);

			% Reshape to #pixels x #examples
			images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
			% Convert to double and rescale to [0,1]
			images = double(images) / 255;

		end
		
		function labels = loadMNISTLabels(filename)
		%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
		%the labels for the MNIST images

			fp = fopen(filename, 'rb');
			assert(fp ~= -1, ['Could not open ', filename, '']);

			magic = fread(fp, 1, 'int32', 0, 'ieee-be');
			assert(magic == 2049, ['Bad magic number in ', filename, '']);

			numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

			labels = fread(fp, inf, 'unsigned char');

			assert(size(labels,1) == numLabels, 'Mismatch in label count');

			fclose(fp);

		end
		
		function [bestModels] = forwardSelection (X)
		% *Forward Selection Algorithm*
		% Inputs: X: The features matrix without the concatenated ones vector along
		% with the last column containing the solution vector
		% Output: A cell with a total of 'n' arrays, each array containing a model
		% which was calculated to be the best in it's category.


			y = X(:,end);
			X(:,end) = [];
			[n, totalFeatures] = size(X);
			loopX = [1:totalFeatures];
			bestModels = cell(1,totalFeatures); % A cell with multiple arrays
			minCost = Inf;
			for i = 1:totalFeatures
				for j = 1:size(loopX,2)
					if i == 1
						currentModel = loopX(j);
					else
						currentModel = [bestModels{i-1} loopX(j)]; 
					end
					curX = X(:,currentModel);
					mdl = fitlm(curX,y);
					curB = mdl.Coefficients.Estimate;
					curCost = ExTwoFunctions.mseMultiFeature(curX,y,curB);
					if curCost < minCost
						minCost = curCost;
						bestModels{i} = currentModel;
					end
				end
				newAddedIndex = bestModels{i}(1,end);
				loopX = loopX(loopX ~= newAddedIndex);
				minCost = Inf;
			end
		end
		
		function [testError] = kFoldCrossValidation(X,k)
		% *K-Fold Cross Validation*
		% Inputs:
		%
		% # X: The features matrix (without the concatenated ones) combined with
		% the solution in the last column
		% # k: 'k' value for k-fold
		%
		% Outputs: MSE

			% Shuffling the test set
			X = X(randperm(size(X,1)),:);
			[featCount,~] = size(X);
			flag = true; % If perfectly divisible
			
			if (size(X,1)/k < 1)
				error('Cannot divide features as k is more than total sample sets');
				testError = -1;
				return;
			end
				
			if rem(featCount,k) == 0
				[R,C] = size(X);
				n = R/k;
				% Converting 2-D matrix to 3-D according to "k"
				% Below line taken from : https://stackoverflow.com/questions/1390909/split-long-2d-matrix-into-the-third-dimension#1390941
				X = permute(reshape(X',[C k R/k]),[2 1 3]);
			else
				flag = false;
				n = ceil(featCount/k);
				xResidue = X(k*(n-1)+1:end,:);
				X(k*(n-1)+1:end,:) = [];
				[R,C] = size(X);
				X = permute(reshape(X',[C k R/k]),[2 1 3]);
				n = n-1;
			end
			testError = 0;
			for i = 1:n
				xLoop = X;
				if n==1 && flag == true
					error('Error: Only one step. Cannot separate testing and training');
					testError = -1;
					return;
				end
				
				if n==1 && flag == false
					xFlat = xResidue(:,1:end-1);
					yFlat = xResidue(:,end);
					testFlatX = X(:,1:end-1);
					testFlatY = X(:,end);
					
				else
					xLoop(:,:,i) = [];
					testFlatX = X(:,:,i);
					testFlatY = testFlatX(:,end);
					testFlatX(:,end) = [];
					xFlat = permute(xLoop,[2,1,3]);
					xFlat = xFlat(:,:);
					xFlat = xFlat';
					
					if flag == false
						xFlat = [xFlat; xResidue];
					end
					
					yFlat = xFlat(:,end);
					xFlat(:,end) = [];
				end  
				
				mdl = fitlm(xFlat,yFlat);
				mse = ExTwoFunctions.mseMultiFeature(testFlatX, testFlatY, mdl.Coefficients.Estimate);
				testError = testError + mse;
			end
			
			if flag == false
				xLoop = X;
				testFlatX = xResidue(:,1:end-1);
				testFlatY = xResidue(:,end);
				xFlat = permute(xLoop,[2,1,3]);
				xFlat = xFlat(:,:);
				xFlat = xFlat';
				yFlat = xFlat(:,end);
				xFlat(:,end) = [];
				mdl = fitlm(xFlat,yFlat);
				mse = ExTwoFunctions.mseMultiFeature(testFlatX, testFlatY, mdl.Coefficients.Estimate);
				testError = testError + mse;
			end
		end
		
		function [jVal] = costLogisticReg (b, X, y, lambda)
		% *Cost function for Regularization of Logistic Regression*
		% Inputs:
		% # b: The beta value
		% # X: The features matrix without ones vector
		% # y: The solution vector
		% # lambda: The regularization constant
		%
		% Output:
		%
		% # jVal: The cost

			[n, ~] = size(X);
			X = [ones(n,1) X];
			mulVal = X * b;
			beta_reg = b(2:end);
			jVal = ((-1/n)*(y'*log(ExTwoFunctions.sigmoid(mulVal))+(1-y)'*log(1-ExTwoFunctions.sigmoid(mulVal)))) + lambda/(2*n)*[beta_reg'*beta_reg];
		end
		
		function [b,costArray, i] = gradientDescentReg (X,y,a,lambda)
		% *GRADIENT DESCENT FOR LOGISTIC REGRESSION*
		%
		% Inputs:
		%
		% # X: The features array combined without ones vector in the first column.
		% # y: The solutions label.

			% Feature Count will have the total number of features + 1, as it has a
			% ones vector concatenated in the first column
		 
			[n, featureCount] = size(X);
			X = [ones(n,1) X];
			% Initializing variables for the loop
			bNew = zeros(featureCount+1, 1);
			loopCondition = true;
			i = 1;
			xTemp = X(:,2:end);
			newCost = ExTwoFunctions.costLogisticReg(bNew,xTemp,y,lambda);
			costArray(1,:) = [1, newCost];
			while loopCondition
				i = i+1;
				bOld = bNew;
				prevCost = newCost;
				beta_reg = bOld(2:end);
				bNew = bOld - a*((1/n)*(X')*((ExTwoFunctions.sigmoid(X*bOld))-y) + lambda/n*[0; beta_reg]);
				xTemp = X(:,2:end);
				newCost = ExTwoFunctions.costLogisticReg(bNew,xTemp,y,lambda);
				
				costArray(i,:) = [i, newCost];
				if (prevCost - newCost) < 0.0000005
					loopCondition = false;
		%             ok = 5
				end
				
		%          a = a+a/9000000;
		%              if rem(i,1000) == 0                 
		%                  i
		%                  newCost
		%              end
		% %         
		%           if i == 300000
		%               loopCondition = false;
		%               ok = 6
		%           end
			 end
			
			b = bNew;
        end
        
        function out = mapFeatureTwo (X,d)
        % A Map-Feature like function for exercise 3
			for i=0:d
				out(:,i+1) = X.^i;
			end
		end
		
		function out = mapGraph (X,d)
			% A Map-Feature like function for exercise 3
				% for i=0:d
				% 	out(i+1) = X.^i;
				% end
				x(:,1) = X;
				x = repmat(x,1,d+1);
				p = repmat([0:d],size(x,1),1);
				out = sum([x.^p], 2);
			end
		
    end
end
        
       
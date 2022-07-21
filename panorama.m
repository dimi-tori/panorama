% Description:
% Creates a panorama stitching of two images.
% If the resulting image doesn't look right, try changing the matching
% accuracy and radius, or use a different pair of images.
%
% Author: Dimitrios Taleporos
%
%run('*/vlfeat-0.9.21/toolbox/vl_setup') % Change this to your vlfeat directory
img1C = imread('up.jpg');       % The first image remains unchanged
img2C = imread('down.jpg');     % The second image is transformed to be mapped onto the first
accuracy = 40;  % How closely two points between the images must match.
% A smaller number is more strict, but at least 4 matching points must be
% found to compute the homography transformation.

img1 = rgb2gray(img1C);
img2 = rgb2gray(img2C);
img1R = img1C(:,:,1);
img1G = img1C(:,:,2);
img1B = img1C(:,:,3);
img2R = img2C(:,:,1);
img2G = img2C(:,:,2);
img2B = img2C(:,:,3);

% Computing SIFT features and distance (takes a while to run)
[f1,d1] = vl_sift(single(img1));
[f2,d2] = vl_sift(single(img2));
dist = dist2(single(d1)',single(d2)');
dist = dist.^0.5;

% Prune results to find the closest matching points
matches = find(dist < accuracy);    % Change accuracy for more or less results
f1Cols = mod(matches,size(d1,2));   % First image
f2Cols = ceil(matches/size(d1,2));  % Second image

% Extract xy coordinates from pruned correspondence points
x1 = zeros(size(matches));
x2 = zeros(size(matches));
y1 = zeros(size(matches));
y2 = zeros(size(matches));

for i = 1:size(matches,1)
   x1(i) = f2(1,f2Cols(i));
   y1(i) = f2(2,f2Cols(i));
   x2(i) = f1(1,f1Cols(i));
   y2(i) = f1(2,f1Cols(i));
end
%% RANSAC implementation
N = 10000;  % Number of iterations
r = 2;      % Matching radius
bestInliers = 0;
for i = 1:N
    % Calculate the homography transformation with 4 random correspondence samples
    p = randsample(size(matches,1),4);
    
    % Insert homography
    A = [x1(p(1)) y1(p(1)) 1 0 0 0 -x1(p(1))*x2(p(1)) -y1(p(1))*x2(p(1)) -x2(p(1));
        0 0 0 x1(p(1)) y1(p(1)) 1 -x1(p(1))*y2(p(1)) -y1(p(1))*y2(p(1)) -y2(p(1));
        x1(p(2)) y1(p(2)) 1 0 0 0 -x1(p(2))*x2(p(2)) -y1(p(2))*x2(p(2)) -x2(p(2));
        0 0 0 x1(p(2)) y1(p(2)) 1 -x1(p(2))*y2(p(2)) -y1(p(2))*y2(p(2)) -y2(p(2));
        x1(p(3)) y1(p(3)) 1 0 0 0 -x1(p(3))*x2(p(3)) -y1(p(3))*x2(p(3)) -x2(p(3));
        0 0 0 x1(p(3)) y1(p(3)) 1 -x1(p(3))*y2(p(3)) -y1(p(3))*y2(p(3)) -y2(p(3));
        x1(p(4)) y1(p(4)) 1 0 0 0 -x1(p(4))*x2(p(4)) -y1(p(4))*x2(p(4)) -x2(p(4));
        0 0 0 x1(p(4)) y1(p(4)) 1 -x1(p(4))*y2(p(4)) -y1(p(4))*y2(p(4)) -y2(p(4))];
    
    [U,S,V]=svd(A); X = V(:,end);
    H = [X(1) X(2) X(3);
        X(4) X(5) X(6);
        X(7) X(8) X(9)];
    
    % Go through each correspondance pair to test the transform validity
    inliers = 0;
    for j = 1:size(matches,1)
       estx = (H(1,1)*x1(j) + H(1,2)*y1(j) + H(1,3))/(H(3,1)*x1(j) + H(3,2)*y1(j) + H(3,3));
       esty = (H(2,1)*x1(j) + H(2,2)*y1(j) + H(2,3))/(H(3,1)*x1(j) + H(3,2)*y1(j) + H(3,3));
       choice1 = 0;
       choice2 = 0;
       
       if ((estx >= x2(j)-r) && (estx <= x2(j)+r))
           choice1 = 1;
       end
       
       if ((esty >= y2(j)-r) && (esty <= y2(j)+r))
           choice2 = 1;
       end
       
       if ((choice1 == 1)&&(choice2 == 1))
           inliers = inliers + 1;
       end
    end
    
    % Keep the best transform
    if (inliers > bestInliers)
        bestInliers = inliers;
        best = H;
    end
end

%% Image Stitching
T = maketform('projective',best');
tR = imtransform(img2R,T);
tG = imtransform(img2G,T);
tB = imtransform(img2B,T);

% Create the combined image that's large enough to fit both images
combinedR = uint8(zeros((size(img1,1)+size(img2,1)),(size(img1,2)+size(img2,2))));
combinedG = combinedR;
combinedB = combinedR;

% Find the position of the second image relative to the first
x = [1 1 size(img1,2) size(img1,2)];
y = [1 size(img1,1) size(img1,1) 1];
tx = (best(1,1)*x + best(1,2)*y + best(1,3))./(best(3,1)*x + best(3,2)*y + best(3,3));
ty = (best(2,1)*x + best(2,2)*y + best(2,3))./(best(3,1)*x + best(3,2)*y + best(3,3));
movex = floor(min(tx));
movey = floor(min(ty));

% Offset in case an image is plotted outside of the array bounds
offsety = 1;
offsetx = 1;
if (movey < 1)
    offsety = 1-movey;
end
if (movex < 1)
    offsetx = 1-movex;
end

% Map second image onto combined image
combinedR(offsety+movey : offsety+movey+size(tR,1)-1, offsetx+movex : offsetx+movex+size(tR,2)-1) = tR;
combinedG(offsety+movey : offsety+movey+size(tR,1)-1, offsetx+movex : offsetx+movex+size(tR,2)-1) = tG;
combinedB(offsety+movey : offsety+movey+size(tR,1)-1, offsetx+movex : offsetx+movex+size(tR,2)-1) = tB;

% Map first image onto combined image
combinedR(offsety : offsety+size(img1,1)-1, offsetx : offsetx+size(img1,2)-1) = max(combinedR(offsety : offsety+size(img1,1)-1, offsetx : offsetx+size(img1,2)-1),img1R);
combinedG(offsety : offsety+size(img1,1)-1, offsetx : offsetx+size(img1,2)-1) = max(combinedG(offsety : offsety+size(img1,1)-1, offsetx : offsetx+size(img1,2)-1),img1G);
combinedB(offsety : offsety+size(img1,1)-1, offsetx : offsetx+size(img1,2)-1) = max(combinedB(offsety : offsety+size(img1,1)-1, offsetx : offsetx+size(img1,2)-1),img1B);

% Display stitched image
figure;
combined = cat(3, combinedR, combinedG, combinedB);
combined( ~any(combinedR,2), :,: ) = [];  %remove zero-padded rows
combined( :, ~any(combinedR,1),: ) = [];  %remove zero-padded columns
imshow(combined)
title('Stitched image')

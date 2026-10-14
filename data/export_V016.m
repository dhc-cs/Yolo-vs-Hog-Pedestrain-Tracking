%imort the provided toolbox 
addpath(genpath('toolbox-master'));

% input and output files
videoFile = 'V016.seq'; % sequence file that has all the frames 
annotationFile = fullfile('candidate_labels', 'V016.vbb');
exportDir = 'V016_export'; % folder that holds frames and labels

%load the video info 
videoInfo = seqIo(videoFile, 'getInfo'); % get video info
data = load(annotationFile, '-mat'); % load the annotation file
annotation = data.A; %assign annotation to A

% create the output folder 
mkdir(exportDir);
framesDir = fullfile(exportDir, 'frames'); % holds the frames

%extract every frame as images 
seqIo(videoFile, 'toImgs', framesDir, 1, ...
    0, videoInfo.numFrames - 1);

%extract annotations 
labelRows = cell(0, 12); %holds the labels

%loop through each frame and get objects
for f = 1:annotation.nFrame
    objects = annotation.objLists{f}; %all the ojbects in the frame
    for k = 1:numel(objects)
        obj = objects(k); %get the object
        box = obj.pos; %get the box position

        visibleBox = [0 0 0 0]; %initialize an empty box
        if isfield(obj, 'posv')
            visibleBox = obj.posv; %get box position it exits
        end

        %add the annotations to the rows
        labelRows(end + 1, :) = { ...
            f - 1, obj.id, annotation.objLbl{obj.id}, ...
            box(1), box(2), box(3), box(4), obj.occl, ...
            visibleBox(1), visibleBox(2), ...
            visibleBox(3), visibleBox(4)};
    end
end

%save the csv file 
labels = cell2table(labelRows, 'VariableNames', { ...
    'Frame', 'ID', 'Label', 'X', 'Y', 'Width', 'Height', ...
    'Occluded', 'VisibleX', 'VisibleY', ...
    'VisibleWidth', 'VisibleHeight'});
% write the table to the csv file
writetable(labels, fullfile(exportDir, 'labels.csv'), 'WriteRowNames', false);

%save the metadata
metadata = table( ...
    videoInfo.numFrames, videoInfo.width, ...
    videoInfo.height, videoInfo.fps, ...
    'VariableNames', {'FrameCount', 'Width', 'Height', 'FPS'});
%again write the table to the csv file
writetable(metadata, fullfile(exportDir, 'metadata.csv'));




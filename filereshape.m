%new code to reshape drawn_data:
myDir = pwd + "/experiments/scoop_seed_04/";
myFiles = dir(fullfile(myDir,"*.mat"));
for k = 1:length(myFiles)
    base = myFiles(k).name;
    filename = myDir + base;
    file = load(filename);
    cell_arr = {};
    drawn_data = file.seg{1,1}.drawn_data;
    if file.seg{1,1}.num_traj ~= 1
        for i = 1:size(drawn_data,2)
            display(drawn_data(:,i));
            for j = 1:size(drawn_data,1)
                drawn_data{j,i} = transpose(drawn_data{j,i});
            end
            col = cell2mat(drawn_data(:,i));
            cell_arr{end+1} = col;
        end
        
    else
        col = transpose(drawn_data);
        cell_arr{end+1} = col;
    end
    
    seg.drawn_data = cell_arr;
    seg.Data = file.seg{1,1}.Data;
    seg.Data_sh = file.seg{1,1}.Data_sh;
    seg.att = file.seg{1,1}.att;
    seg.x0_all = file.seg{1,1}.x0_all;
    seg.dt = file.seg{1,1}.dt;
    save(filename,"seg");
end
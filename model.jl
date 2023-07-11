using StatsBase, Random, Distributions

# define an agent and all agent properties
Base.@kwdef mutable struct Human
    idx::Int64 = 0 
    agegroup::Int64 = 0    # in years. don't really need this but left it incase needed later
    comorbidity::Int64 = 0 # 0, 1=(1 - 3 comorbidity), 2=(4+ comorbidity)
    rsvtype::Int64 = 0 # 1 = outpatient, 2 = emergency, 3 = hospital
    rsvmonth::Int64 = 0  # from months 1 to 9 
    rsvicu::Bool = false # 1 = yes, 0 = no
    rsvmv::Bool = false # 1 = yes, 0 = no
end
Base.show(io::IO, ::MIME"text/plain", z::Human) = dump(z)

## system parameters
Base.@kwdef mutable struct ModelParameters    ## use @with_kw from Parameters
    popsize::Int64 = 100000
    modeltime::Int64 = 300
    numofsims::Int64 = 1000
    usapopulation = 78_913_275
end

# constant variables
const humans = Array{Human}(undef, 0) 
const p = ModelParameters()  ## setup default parameters
pc(x) = Int(round(x / p.usapopulation * p.popsize)) # convert to per-capita

#l# Iniltialization Functions 
reset_params() = reset_params(ModelParameters())
function reset_params(ip::ModelParameters)
    # the p is a global const
    # the ip is an incoming different instance of parameters 
    # copy the values from ip to p. 
    ip.popsize == 0 && error("no population size given")
    for x in propertynames(p)
        setfield!(p, x, getfield(ip, x))
    end
    # resize the human array to change population size
    resize!(humans, p.popsize)
end

function get_agegroup(x::Human) 
    # helper function to printout age group 
    if x.agegroup == 1 
        return "60 - 64"
    elseif x.agegroup == 2
        return "65 - 69"
    elseif x.agegroup == 3
        return "70 - 74"
    elseif x.agegroup == 4
        return "75 - 79"
    elseif x.agegroup == 5
        return "80 - 84"
    elseif x.agegroup == 6
        return "85+"
    end
end

function initialize() 
    # population table from census
    # https://www2.census.gov/programs-surveys/popest/datasets/2020-2022/national/asrh/
    pop_distribution = pc.([21118423, 18631422, 15157017, 10861000, 6659545, 6485868])
    sf_agegroups = shuffle!(inverse_rle([1, 2, 3, 4, 5, 6], pop_distribution)) #
    @info "pop size per capita (from census data):" pop_distribution, sum(pop_distribution)

    # data for the number of comorbidity per age group
    co1 = Categorical([32, 58.2, 9.8] ./ 100)
    co2 = Categorical([26.5, 62.5, 11.0] ./ 100)
    co3 = Categorical([21.5, 65.0, 13.5] ./ 100)
    co4 = Categorical([17.0, 67.5, 15.5] ./ 100)
    co5 = Categorical([14.5, 67.5, 18.0] ./ 100)
    co6 = Categorical([11.5, 66.5, 22.0] ./ 100)
    co = [co1, co2, co3, co4, co5, co6]
    
    for i = 1:length(humans)
        humans[i] = Human() 
        x = humans[i] 
        x.idx = i  
        x.agegroup = sf_agegroups[i]
        x.comorbidity = rand(co[x.agegroup]) - 1 # since categorical is 1-based. 
    end
end

function incidence() 
    # sample the annual incidence 
    #   q: if this is annual incidence, then can we use it over 7 months? 
    #   q: 
    inc_outpatient = rand(Uniform(1595, 2669))
    inc_emergency = rand(Uniform(23, 387)) 
    inc_hospital = rand(Uniform(178, 250)) * transpose([0.055, 0.782, 0.163]) # split hospitalization over comorbid peopl
    incidence = [inc_outpatient, inc_emergency, inc_hospital...]
    @info "sampled incidence" incidence
   
    # split over months
    incidence_per_month = Int.(round.([0.0017, 0.0165, 0.0449, 0.1649, 0.3047, 0.2365, 0.1660, 0.0591, 0.0057] .* transpose(incidence)))
    @info "total incidence: " sum(incidence_per_month) sum(incidence_per_month, dims=1) 

    outpatient_months = inverse_rle([1, 2, 3, 4, 5, 6, 7, 8, 9],  incidence_per_month[:, 1])
    total_outpatients = length(outpatient_months)
    non_sick_humans = humans[findall(x -> x.rsvmonth == 0, humans)[1:total_outpatients]] # don't need to sample here since the `initialize` function already shuffled the population
    for i in eachindex(outpatient_months)
        non_sick_humans[i].rsvmonth = outpatient_months[i] 
        non_sick_humans[i].rsvtype = 1
    end 

    # repeat the same for emergency
    emergency_months = inverse_rle([1, 2, 3, 4, 5, 6, 7, 8, 9],  incidence_per_month[:, 2])
    total_emergencies = length(emergency_months)
    non_sick_humans = humans[findall(x -> x.rsvmonth == 0, humans)[1:total_emergencies]] # don't need to sample here since the `initialize` function already shuffled the population
    for i in eachindex(emergency_months)
        non_sick_humans[i].rsvmonth = emergency_months[i] 
        non_sick_humans[i].rsvtype = 2
    end

    # repeat the same for hospitalization (no comorbidity)
    hospital_months_c1 = inverse_rle([1, 2, 3, 4, 5, 6, 7, 8, 9],  incidence_per_month[:, 3])
    total_hospitals = length(hospital_months_c1)
    total_icu = Int(round(total_hospitals * 0.24))
    non_sick_humans = humans[findall(x -> x.rsvmonth == 0 && x.comorbidity == 0, humans)[1:total_hospitals]] # don't need to sample here since the `initialize` function already shuffled the population
    for i in eachindex(hospital_months_c1)
        non_sick_humans[i].rsvmonth = hospital_months_c1[i] 
        non_sick_humans[i].rsvtype = 3
        if total_icu > 0
            non_sick_humans[i].rsvicu = true
            total_icu -= 1
        end
    end

    # repeat the same for hospitalization (1-3 comorbidity)
    hospital_months_c2 = inverse_rle([1, 2, 3, 4, 5, 6, 7, 8, 9],  incidence_per_month[:, 4])
    total_hospitals = length(hospital_months_c2)
    total_icu = Int(round(total_hospitals * 0.15))
    non_sick_humans = humans[findall(x -> x.rsvmonth == 0 && x.comorbidity == 1, humans)[1:total_hospitals]] # don't need to sample here since the `initialize` function already shuffled the population
    for i in eachindex(hospital_months_c2)
        non_sick_humans[i].rsvmonth = hospital_months_c2[i] 
        non_sick_humans[i].rsvtype = 3
        if total_icu > 0 
            non_sick_humans[i].rsvicu = true 
            total_icu -= 1
        end
    end

    # repeat the same for hospitalization (4+ comorbidity)
    hospital_months_c3 = inverse_rle([1, 2, 3, 4, 5, 6, 7, 8, 9],  incidence_per_month[:, 5])
    total_hospitals = length(hospital_months_c3)
    total_icu = Int(round(total_hospitals * 0.12))
    non_sick_humans = humans[findall(x -> x.rsvmonth == 0 && x.comorbidity == 2, humans)[1:total_hospitals]] # don't need to sample here since the `initialize` function already shuffled the population
    for i in eachindex(hospital_months_c3)
        non_sick_humans[i].rsvmonth = hospital_months_c3[i] 
        non_sick_humans[i].rsvtype = 3
        if total_icu > 0 
            non_sick_humans[i].rsvicu = true 
            total_icu -= 1
        end
    end

    # MV 
    mv_elg = findall(x -> x.rsvicu == true, humans) 
    mv_tot = Int(round(length(mv_elg) * 0.166)) 
    for x in humans[mv_elg[1:mv_tot]] 
        x.rsvmv = true
    end 

    incidence_check()
end

function incidence_check()
    # check 
    all_sick = length(findall(x -> x.rsvtype != 0, humans))   
    outpatients = length(findall(x -> x.rsvtype == 1, humans))   
    emergency = length(findall(x -> x.rsvtype == 2, humans))   
    hosp = length(findall(x -> x.rsvtype == 3, humans))
    hosp_no_comorbid  = length(findall(x -> x.rsvtype == 3 && x.comorbidity == 0, humans))
    hosp_13_comorbid = length(findall(x -> x.rsvtype == 3 && x.comorbidity == 1, humans))
    hosp_4_comorbid = length(findall(x -> x.rsvtype == 3 && x.comorbidity == 2, humans))
    icu_no_comorbid  = length(findall(x -> x.rsvtype == 3 && x.comorbidity == 0 && x.rsvicu == true, humans))
    icu_13_comorbid = length(findall(x -> x.rsvtype == 3 && x.comorbidity == 1 && x.rsvicu == true, humans))
    icu_4_comorbid = length(findall(x -> x.rsvtype == 3 && x.comorbidity == 2 && x.rsvicu == true, humans))
    @info "total humans sick " all_sick outpatients emergency hosp hosp_no_comorbid  hosp_13_comorbid hosp_4_comorbid icu_no_comorbid icu_13_comorbid icu_4_comorbid
end

function vaccine() 
    # julia> [0.826 - 0.00757143*i for i = 0:7]
    # 8-element Vector{Float64}:
    #  0.826
    #  0.8184285699999999
    #  0.81085714
    #  0.8032857099999999
    #  0.79571428
    #  0.78814285
    #  0.78057142
    #  0.77299999

    # vaccine must be incremental --- how to do?

end


using StatsBase, Random, Distributions, DataFrames, Logging, CSV, ProgressBars

@enum INFTYPE SUSC=0 NMA=1 OP=2 ED=3 HOSP=4 ICU=5 MV=6 DEATH=7
@enum VAXTYPE GSK=1 PFI=2 PFIGSK=3
@enum VAXMONTH SEP=1 OCT=2

# define an agent and all agent properties
Base.@kwdef mutable struct Human
    idx::Int64 = 0 
    agegroup::Int64 = 0    # in years. don't really need this but left it incase needed later
    comorbidity::Int64 = 0 # 0, 1=(1 - 3 comorbidity), 2=(4+ comorbidity)
    rsvtype::INFTYPE = SUSC # 1 = outpatient, 2 = emergency, 3 = hospital
    rsvdays::Dict{String, Float64} = Dict("nma" => 0.0, "symp" => 0.0, "hosp" => 0.0, "icu" => 0.0, "mv" => 0.0)
    rsvmonth::Int64 = 0 
    vaccinated::Bool = false 
    vaxmonth::VAXMONTH = SEP  # default
    vaxtype::VAXTYPE = GSK # 1 
    vaxeff_op::Vector{Float64} = zeros(Float64, 24)
    vaxeff_ip::Vector{Float64} = zeros(Float64, 24)
end
Base.show(io::IO, ::MIME"text/plain", z::Human) = dump(z)

## system parameters
Base.@kwdef mutable struct ModelParameters    ## use @with_kw from Parameters
    popsize::Int64 = 100000
    numofsims::Int64 = 1000
    usapopulation = 78_913_275
    vaccine_scenario::VAXTYPE = GSK # 0 - gsk, 1 pfizer 
    vaccine_coverage::Float64 = 1.0 
    incidence_mth_distribution::Vector{Float64} = [0.0017, 0.0165, 0.0449, 0.1649, 0.3047, 0.2365, 0.1660, 0.0591, 0.0057]
    current_season::Int64 = 1 # either running 1 season or 2 seasons 
end

# constant variables
const humans = Array{Human}(undef, 0) 
const p = ModelParameters()  ## setup default parameters
pc(x) = Int(round(x / p.usapopulation * p.popsize)) # convert to per-capita

function simulations()
    # redirect the @info statements
    logger = NullLogger()
    global_logger(logger)

   # reset_params() 
    Random.seed!(53)

    # vaccine_scenario
    p.vaccine_scenario = PFIGSK
    p.vaccine_coverage = 0.66
    fname = string(p.vaccine_scenario)
    fname_coverage = "$(string(p.vaccine_coverage * 100))" # edit these if changing the parameters

    names = ["simid", "scenario", "num_gsk", "num_pfi", "total_sick", "nma", "outpatients", "ed", "totalhosp", "gw", "icu", "mv", "deaths", "nmadays", "sympdays", "hospdays",  "icu_nmv", "icu_mv", "totalqalys", "qalyslost"]
    df_novax = DataFrame([name => [] for name in names])
    df_wivax = DataFrame([name => [] for name in names])
    
    for i in ProgressBar(1:1000)
        #@info "simulation $i"
        initialize() # initialize the population plus assign vaccine efficacy
    
        p.current_season = 1 # set the default season 

        # without vaccine scenarios 
        for ssn in (1, 2) 
            p.current_season = ssn
            op_and_ed()       # incidence
            hosp_and_icu()    # hospitalization and icu  
        end
        death()
        sample_days()     # sample the days spent in each infection category 
        
        _empty_vax = zeros(7, 2) # to be consistent with the with vax dataframe
        _data_incidence = collect_incidence() 
        _data_infdays = collect_days() 
        _data_qalys = qalys() 
        _data = hcat(_empty_vax, _data_incidence, _data_infdays, _data_qalys)
        map(x -> push!(df_novax, (i, "novax", x...)), eachrow(_data))

        # what happens with vaccine?
        vaccine() 
        _dvax = collect_vaccines()
        _data_incidence = collect_incidence() 
        _data_infdays = collect_days() 
        _data_qalys = qalys() 
        _data = hcat(_dvax, _data_incidence, _data_infdays, _data_qalys)
        map(x -> push!(df_wivax, (i, "wivax", x...)), eachrow(_data))

    end
    CSV.write("$(fname)_$(fname_coverage)_novaccine.csv", df_novax)
    CSV.write("$(fname)_$(fname_coverage)_wivaccine.csv", df_wivax)

    df_novax, df_wivax
end

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

function roll_vaccine(month, efficacy)
    if month != 1 && month != 2 
        error("can't roll vaccine efficacy") 
    end 
    a = circshift(efficacy, month)
    a[1:month] .= 0
    return a
end

function apply_vaccine(x::Human, mth::VAXMONTH) 
    # old numbers: linearly interpolate
    
    #gsk_outpatient = [0.82, 0.82, 0.82, 0.82, 0.82, 0.82, 0.81, 0.80, 0.79]
    #gsk_inpatient = [0.94, 0.93, 0.93, 0.93, 0.92, 0.91, 0.90, 0.89, 0.87]
    #pfi_outpatient = [0.65, 0.65, 0.64, 0.64, 0.63, 0.62, 0.61, 0.60, 0.58]
    #pfi_inpatient = [0.89, 0.89, 0.88, 0.88, 0.88, 0.88, 0.87, 0.86, 0.85]
    
    #gsk_outpatient = [0.826, 0.826, 0.826, 0.826, 0.826, 0.826, 0.826, 0.797, 0.768]
    #gsk_inpatient = [0.941, 0.941, 0.941, 0.941, 0.941, 0.941, 0.941, 0.912, 0.883]
    #pfi_outpatient = [0.651, 0.651, 0.651, 0.651, 0.651, 0.651, 0.651, 0.651, 0.619]
    #pfi_inpatient = [0.889, 0.889, 0.889, 0.889, 0.889, 0.889, 0.889, 0.889, 0.868]

    # FIXED VE
    #gsk_outpatient = [0.826, 0.826, 0.826, 0.826, 0.826, 0.826, 0.826, 0.797, 0.768, 0.739, 0.710, 0.681, 0.672, 0.672, 0.672, 0.672, 0.672, 0.672, 0.560, 0.448, 0.336, 0.224, 0.112, 0.000]
    #gsk_inpatient = [0.941, 0.941, 0.941, 0.941, 0.941, 0.941, 0.941, 0.912, 0.883, 0.854, 0.826, 0.797, 0.788, 0.788, 0.788, 0.788, 0.788, 0.788, 0.657, 0.525, 0.394, 0.263, 0.131, 0.000]
    #pfi_outpatient = [0.651, 0.651, 0.651, 0.651, 0.651, 0.651, 0.651, 0.651, 0.619, 0.586, 0.554, 0.521, 0.489, 0.489, 0.489, 0.489, 0.489, 0.489, 0.408, 0.326, 0.245, 0.163, 0.082, 0.000]
    #pfi_inpatient = [0.889, 0.889, 0.889, 0.889, 0.889, 0.889, 0.889, 0.889, 0.868, 0.848, 0.827, 0.786, 0.786, 0.786, 0.786, 0.786, 0.786, 0.786, 0.655, 0.524, 0.393, 0.262, 0.131, 0.000]

    # VE
    gsk_outpatient = [0.82, 0.82, 0.82, 0.82, 0.82, 0.82, 0.81, 0.80, 0.79, 0.78, 0.76, 0.73, 0.69, 0.64, 0.57, 0.49, 0.40, 0.32, 0.23, 0.16, 0.10, 0.06, 0.03, 0.00]
    gsk_inpatient = [0.94, 0.93, 0.93, 0.93, 0.92, 0.91, 0.90, 0.89, 0.87, 0.85, 0.82, 0.78, 0.73, 0.67, 0.60, 0.52, 0.44, 0.35, 0.27, 0.20, 0.13, 0.08, 0.04, 0.00]
    pfi_outpatient = [0.65, 0.65, 0.64, 0.64, 0.63, 0.62, 0.61, 0.60, 0.58, 0.56, 0.54, 0.51, 0.47, 0.44, 0.39, 0.35, 0.30, 0.25, 0.20, 0.16, 0.11, 0.07, 0.04, 0.01]
    pfi_inpatient = [0.89, 0.89, 0.88, 0.88, 0.88, 0.88, 0.87, 0.86, 0.85, 0.84, 0.82, 0.79, 0.75, 0.70, 0.64, 0.56, 0.48, 0.39, 0.30, 0.22, 0.15, 0.09, 0.04, 0.00]

    if p.vaccine_scenario == PFIGSK 
        _st = rand() < 0.5 ? PFI : GSK
    else 
        _st = p.vaccine_scenario
    end 
    if _st == GSK
        x.vaccinated = true
        x.vaxtype = GSK
        x.vaxeff_op = roll_vaccine(Int(mth), gsk_outpatient)
        x.vaxeff_ip = roll_vaccine(Int(mth), gsk_inpatient)
    elseif _st == PFI
        x.vaccinated = true
        x.vaxtype = PFI
        x.vaxeff_op = roll_vaccine(Int(mth), pfi_outpatient)
        x.vaxeff_ip = roll_vaccine(Int(mth), pfi_inpatient)
    else 
        error("unknown vaccine type")
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

    _vaxelig = Int(round(100000 * p.vaccine_coverage))   
    vaxelig = sample(humans, _vaxelig, replace=false) 
    sepelig = vaxelig[1:(end÷2)]
    octelig = vaxelig[(end÷2 + 1):end]
    for x in sepelig 
        apply_vaccine(x, SEP)
    end 
    for x in octelig 
        apply_vaccine(x, OCT)
    end 
        
    shuffle!(humans) # shuffle the humans for stochastictiy

    # get total vaccinated 
    num_vax_pfi = length(findall(x -> x.vaccinated == true && x.vaxtype == PFI, humans))
    num_vax_gsk = length(findall(x -> x.vaccinated == true && x.vaxtype == GSK, humans))

    return (num_vax_gsk, num_vax_pfi)
end

function mth_indices() 
    # either returns [1, 2, 3, 4, 5, 6, 7, 8, 9] 
    # or [13, 14, 15, 16, 17, 18, 19, 20, 21]
    month_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9] .+ (12 * (p.current_season - 1))
end

function hosp_and_icu()
    hosp_age_distr = [0.062, 0.126, 0.265, 0.548] # distribution of hospitalization by agegroups 1, (2,3), (4,5), and 6
    hosp_comorbid_distr = [0.055, 0.782, 0.163] # distribution over comorbidity (0 = 0 comorbidity, 1 = 1-3 comorbidity, 2 = 2 comorbidity) 
    icu_comorbid_prob = [0.24, 0.15, 0.12]

    inc_hospital = rand(Uniform(178, 250)) * hosp_comorbid_distr # split hospitalization over comorbid peopl
    #@info "incidence hospitalization" inc_hospital
    incidence_per_month = round.(Int, p.incidence_mth_distribution .* transpose(inc_hospital))
    incidence_co0 = incidence_per_month[:, 1]
    incidence_co13 = incidence_per_month[:, 2]
    incidence_co4 = incidence_per_month[:, 3] 
    #incidence_allco = [incidence_co0, incidence_co13, incidence_co4]

    for co in (0, 1, 2) # go through each comorbidity category (0 = 0 comorbidity, 1 = 1-3 comorbidity, 2 = 2 comorbidity)
        hosp_by_month = incidence_per_month[:, (co+1)] # since arg is 0 based
        total_hosp = sum(hosp_by_month)  # total hospitalization (found by summing over the 9 , could also use the sampled number but thats not rounded)
        hosp_by_age = round.(Int, hosp_age_distr .* total_hosp) # split the total hospitalization into 4 agegroups
        hosp_by_age[1] = hosp_by_age[1] + (total_hosp - sum(hosp_by_age)) # rounding can cause off by 1, add the difference to the first age group (both + and negative are taken into account here)
        #@assert sum(hosp_by_age) == total_hosp # sanity check
    
        # lets find susc individuals in each group, stratified by comorbidity 
        # we don't need ALL the incideces, only the total counts in hosp_by_age
        h_ag1 = findall(x -> x.agegroup == 1 && x.rsvtype == SUSC && x.comorbidity == co, humans)[1:hosp_by_age[1]]
        h_ag2 = findall(x -> x.agegroup in (2, 3) && x.rsvtype == SUSC && x.comorbidity == co, humans)[1:hosp_by_age[2]]
        h_ag3 = findall(x -> x.agegroup in (4, 5) && x.rsvtype == SUSC && x.comorbidity == co, humans)[1:hosp_by_age[3]]
        h_ag4 = findall(x -> x.agegroup == 6 && x.rsvtype == SUSC && x.comorbidity == co, humans)[1:hosp_by_age[4]]
        h_ag = shuffle!(vcat(h_ag1, h_ag2, h_ag3, h_ag4)) # shuffle these because we loop through in order (otherwise ag1s all get ICU)

        # lets also get the total icu count
        picu = icu_comorbid_prob[co+1] # since co is 0 based, variable only used once but easier to read
        total_icu = Int(round(total_hosp * picu)) # multiply the total hospitalization by the probability of icu to get total icu count

        # take the hosp_by_month (which are counts over months) and is gauranteed to add to the same as hosp_by_age
        # and create a vector with the month of infections totaling this count of hosp_by_month
        hospital_months_c2 = inverse_rle(mth_indices(),  hosp_by_month)

        # go through each agent (the exact count that needs to be hospitalized)   
        # assign them the right type, the month, and check whether they will be in the icu
        for x in h_ag
            humans[x].rsvmonth = pop!(hospital_months_c2)
            humans[x].rsvtype = HOSP
            # check ICU
            if total_icu > 0 
                humans[x].rsvtype = ICU # overwrite with ICU
                total_icu -= 1
                if rand() < 0.166 
                    humans[x].rsvtype = MV # overwrite with MV
                end
            end
        end        
    end

  
    return sum(inc_hospital)
end

function death() 
    # DEATH LOGIC 
    death_prob = rand(Uniform(0.056, 0.076))  # 1. sample death probability from uniform distribution 
    all_hospitalized = findall(x -> x.rsvtype in (HOSP, ICU, MV), humans) # 2. find all hospitalized (gw, icu, mv)
    total_death_count = round(Int, death_prob * length(all_hospitalized)) # 3. multiply death probability by the number of hospitalized to get a TOTAL NUMBER OF DEATHS 

    # now need to split all those deaths by age groups (age group probs add to 1) 
    death_prob_by_age = [0.26, 0.22, 0.17, 0.35] # 60-69, 70-79, 80-84, 85+]
    total_death_by_agegroup = round.(Int, total_death_count * death_prob_by_age) # this is a 4 element vector where each element corresponds to death in each age group

    # split the total deaths in age groups to hosp/icu (37% of all deaths are in the hosp, 63% in icu/mv)
    deaths_in_hosp = round.(Int, total_death_by_agegroup .* 0.37) # 4 element vector 
    deaths_in_icu = round.(Int, total_death_by_agegroup .* 0.63)  # 4 element vector

    idx_to_die = Int64[]

    # at this point, we need to sample the right people 
    pool_hosp_ag1 = shuffle!(findall(x -> x.rsvtype == HOSP && x.agegroup in (1, 2), humans))
    pool_hosp_ag2 = shuffle!(findall(x -> x.rsvtype == HOSP && x.agegroup in (3, 4), humans))
    pool_hosp_ag3 = shuffle!(findall(x -> x.rsvtype == HOSP && x.agegroup == 5, humans))
    pool_hosp_ag4 = shuffle!(findall(x -> x.rsvtype == HOSP && x.agegroup == 6, humans))
    pool = [pool_hosp_ag1, pool_hosp_ag2, pool_hosp_ag3, pool_hosp_ag4]
    # now we go through each age grou array and find the right number of people to die in each age group
    # we want to make sure that each array, i.e., todie_hosp_ag1, todie_hosp_ag2... have the right number of people to die
    # that is, todie_hosp_ag1 should have atleast deaths_in_hosp[1] 
    # and todie_hosp_ag2 should have atleast deaths_in_hosp[2] and so on
    for i = 1:4
        if length(pool[i]) < deaths_in_hosp[i]
            deaths_in_hosp[i] -= deaths_in_hosp[i] - length(pool[i])        
        end
        # now that the deaths per age group are adjusted to be maximum the the number of people in each age group, we can kill them
        push!(idx_to_die, pool[i][1:deaths_in_hosp[i]]...)
    end

    pool_icumv_ag1 = shuffle!(findall(x -> x.rsvtype in (ICU, MV) && x.agegroup in (1, 2), humans))
    pool_icumv_ag2 = shuffle!(findall(x -> x.rsvtype in (ICU, MV) && x.agegroup in (3, 4), humans))
    pool_icumv_ag3 = shuffle!(findall(x -> x.rsvtype in (ICU, MV) && x.agegroup == 5, humans))
    pool_icumv_ag4 = shuffle!(findall(x -> x.rsvtype in (ICU, MV) && x.agegroup == 6, humans))
    pool_icu = [pool_icumv_ag1, pool_icumv_ag2, pool_icumv_ag3, pool_icumv_ag4]
    for i = 1:4
        #println("i: $i deaths in icu: $(deaths_in_icu[i]), length of pool: $(length(todie[i]))")
        if length(pool_icu[i]) < deaths_in_icu[i] 
            deaths_in_icu[i] -= deaths_in_icu[i] - length(pool_icu[i])
        end
        push!(idx_to_die, pool_icu[i][1:deaths_in_icu[i]]...)
    end

    for x in idx_to_die 
        humans[x].rsvtype = DEATH # overwrite with DEATH
    end
end

function op_and_ed() 
    # sample the annual incidence 
    inc_outpatient = rand(Uniform(1595, 2669))
    inc_emergency = rand(Uniform(23, 387)) 
    inc_hospital = rand(Uniform(178, 250)) * transpose([0.055, 0.782, 0.163]) # split hospitalization over comorbid peopl
    incidence = [inc_outpatient, inc_emergency]
    @info "sampled incidence" incidence
   
    # split over 9 months, creates a matrix with 5 columns 
    # column 1: outpatients over 9 months (september, october, november, december, january, february, march, april, may)
    # column 2: emergency over 9 months 
    # column 3-5: hospitalization over 9 months (split over comorbidity)
    incidence_per_month = Int.(round.(p.incidence_mth_distribution .* transpose(incidence)))
    @info "total incidence and (op, ed, hosp(0comorbid), hosp(2-4co), hosp(4+co)): " sum(incidence_per_month) sum(incidence_per_month, dims=1) 

    # distribute outpatients to the population
    outpatient_months = inverse_rle(mth_indices(),  incidence_per_month[:, 1])
    total_outpatients = length(outpatient_months)
    non_sick_humans = humans[findall(x -> x.rsvtype == SUSC, humans)[1:total_outpatients]] # don't need to sample here since the `initialize` function already shuffled the population
    for i in eachindex(outpatient_months)
        non_sick_humans[i].rsvmonth = outpatient_months[i] 
        non_sick_humans[i].rsvtype = OP
    end

    # repeat the same for emergency
    emergency_months = inverse_rle(mth_indices(),  incidence_per_month[:, 2])
    total_emergencies = length(emergency_months)
    non_sick_humans = humans[findall(x -> x.rsvtype == SUSC, humans)[1:total_emergencies]] # don't need to sample here since the `initialize` function already shuffled the population
    for i in eachindex(emergency_months)
        non_sick_humans[i].rsvmonth = emergency_months[i] 
        non_sick_humans[i].rsvtype = ED
    end

    #  hospitalization
    # _incidence_hospital(incidence_per_month[:, 3], 0) # 0 no comorbidity
    # return _incidence_hospital(incidence_per_month[:, 4], 1) # 1: 1 - 3 comorbidities
    #_incidence_hospital(incidence_per_month[:, 5], 2) # 
end

function sample_days() 
    # add number of days sick to agent object
    all_sick = findall(x -> Int(x.rsvtype) > 0, humans)
    for idx in all_sick
        x = humans[idx]
        sample_inf_days_human(x)
    end
end

function sample_inf_days_human(x::Human) 
    rsvtype = x.rsvtype

    nmadays = 0 
    sympdays = 0 
    hospdays = 0 
    icu_mv_days = 0 
    icu_nmv_days = 0 
    x.rsvdays["nma"] = nmadays
    x.rsvdays["symp"] = sympdays
    x.rsvdays["hosp"] = hospdays
    x.rsvdays["icu"] = icu_nmv_days
    x.rsvdays["mv"] = icu_mv_days

    if rsvtype == NMA 
        nmadays += rand(Uniform(2, 8)) 
    elseif rsvtype == OP 
        sympdays += rand(Uniform(7, 14))
    elseif rsvtype == ED
        sympdays += rand(Uniform(7, 14))
    elseif rsvtype == HOSP 
        sympdays += 4  # 4 days of symptoms before hospital admission
        hospdays += rand(Gamma(1.2258, 5.0582)) # mean 6.2 days
    elseif rsvtype == ICU || rsvtype == MV 
        sympdays += 4  # 4 days of symptoms before hospital admission
        hospdays += 1  # one day general ward before ICU 
        if rsvtype == ICU 
            icu_nmv_days += rand(Gamma(1.5625,2.8802))
        else 
            icu_mv_days += rand(Gamma(1.5625,2.8802)) 
        end
        hospdays += 2  # since person did not die 
    elseif rsvtype == DEATH 
        sympdays += 4  # 4 days of symptoms before hospital admission
        if rand() < 0.37
            hospdays += rand(Gamma(1.2258, 5.0582))  # one day general ward before ICU
        else # ICU
            hospdays += 1
            if rand() < 0.166 
                icu_mv_days += rand(Gamma(1.5625,2.8802)) 
            else 
                icu_nmv_days += rand(Gamma(1.5625,2.8802))
            end
        end
    end
    x.rsvdays["nma"] = nmadays
    x.rsvdays["symp"] = sympdays
    x.rsvdays["hosp"] = hospdays
    x.rsvdays["icu"] = icu_nmv_days
    x.rsvdays["mv"] = icu_mv_days
    return (nmadays, sympdays, hospdays, icu_nmv_days, icu_mv_days)
end

function collect_days() 
    split_data = zeros(Float64, 7, 5) # 7 rows for all + 6 age groups 
    all_sick = findall(x -> Int(x.rsvtype) > 0, humans)
    for idx in all_sick
        x = humans[idx]
        ag = x.agegroup
        nma = x.rsvdays["nma"]
        s = x.rsvdays["symp"]
        h =  x.rsvdays["hosp"] 
        i_nmv = x.rsvdays["icu"]
        i_mv = x.rsvdays["mv"]
        split_data[1, :] .+= [nma, s, h, i_nmv, i_mv]
        split_data[ag+1, :] .+= [nma, s, h, i_nmv, i_mv]
    end
    return split_data 
end

function collect_toi() 
    # function calculates the time of infection for each agent 
    all_sick = findall(x -> Int(x.rsvtype) > 0, humans)
    split_data = zeros(Float64, 7, 1)

    toi = [humans[i].rsvmonth for i in all_sick]
    countmap(toi)
end

function collect_vaccines() 
    split_data = zeros(Float64, 7, 2)
    num_gsk = [length(findall(x -> x.vaccinated == true && x.agegroup == ag && x.vaxtype == GSK, humans)) for ag in 1:6]
    num_pfi = [length(findall(x -> x.vaccinated == true && x.agegroup == ag && x.vaxtype == PFI, humans)) for ag in 1:6]

    split_data[1, :] .= (sum(num_gsk), sum(num_pfi))
    split_data[2:end, :] .= hcat(num_gsk, num_pfi)
    return split_data 
end

function collect_incidence()
    # NOTE: HOSP, ICU, MV, DEATH ARE "SPLIT". That is 
    # technically ICU, MV, DEATH ARE ALL PART OF "HOSPITALIZATION" 
    # SO TOTAL HOSPITALIZATION REQUIRES SUMMING THEM UP
    # --> I HAVE ADDED THIS AS A COLUMN `allhospitalizations`
    
    split_data = zeros(Float64, 7, 9) # 7 rows for all + 6 age groups 
    for ag in 0:6 
        if ag == 0 
            _humans = humans[findall(x -> Int(x.rsvtype) > 0, humans)]
        else 
            _humans = humans[findall(x -> Int(x.rsvtype) > 0 && x.agegroup == ag, humans)]
        end
        all_sick = length(findall(x -> Int(x.rsvtype) > 0, _humans))  
        nma = length(findall(x -> x.rsvtype == NMA, _humans))
        outpatients = length(findall(x -> x.rsvtype == OP, _humans))   
        emergency = length(findall(x -> x.rsvtype == ED, _humans))   
        hosp = length(findall(x -> x.rsvtype == HOSP, _humans))
        totalicu = length(findall(x -> x.rsvtype == ICU, _humans))
        totalmv = length(findall(x -> x.rsvtype == MV, _humans))
        totaldeath = length(findall(x -> x.rsvtype == DEATH, _humans))
        allhospitalizations = hosp + totalicu + totalmv + totaldeath
        split_data[ag+1, :] .= (all_sick, nma, outpatients, emergency, allhospitalizations, hosp, totalicu, totalmv, totaldeath)
    end
    return split_data
end

function vaccine() 
    op_to_nma = 0 
    ip_to_op = 0
    all_sick = findall(x -> Int(x.rsvtype) > 0, humans) # find all sick individuals to calculate their outcomes
    for i in all_sick
        x = humans[i]  
        rn = rand() 
        if x.rsvtype == OP || x.rsvtype == ED  # they become NMA 
            if rn < x.vaxeff_op[x.rsvmonth]
                x.rsvtype = NMA
                op_to_nma += 1
                sample_inf_days_human(x)
            end
        end 
        if x.rsvtype in (HOSP, ICU, MV, DEATH)
            if rn < x.vaxeff_ip[x.rsvmonth]
                x.rsvtype = OP
                ip_to_op += 1
                sample_inf_days_human(x)
            end
        end 
    end
    return (op_to_nma, ip_to_op)
end

function qalys() 
    split_data = zeros(Float64, 7, 2) 

    wgt_norsv = [0.77, 0.76, 0.74, 0.70, 0.63, 0.51]
    wgt_nma = [0.6776, 0.6688, 0.6512, 0.6160, 0.5544, 0.4488]
    wgt_op = [0.5852, 0.5776, 0.5624, 0.532, 0.4788, 0.3876]
    wgt_nonicu = [0.2695, 0.2660, 0.2590, 0.2450, 0.2205, 0.1785]
    wgt_icu = [0.077, 0.076, 0.074, 0.070, 0.063, 0.051]
 
    # go througḣall sick but non-dead people 
    all_sick = findall(x -> Int(x.rsvtype) > 0 && x.rsvtype != DEATH, humans)
    for i in all_sick
        x = humans[i]  

        ag = x.agegroup

        nma_days = x.rsvdays["nma"]
        symp_days = x.rsvdays["symp"]
        hosp_days =  x.rsvdays["hosp"] 
        icu_days = x.rsvdays["icu"]
        mv_days = x.rsvdays["mv"]

        non_rsv_days = 365 - (nma_days + symp_days + hosp_days + icu_days + mv_days)
        totalqaly = wgt_norsv[ag]*non_rsv_days + wgt_nma[ag]*nma_days + wgt_op[ag]*symp_days + wgt_nonicu[ag]*hosp_days + wgt_icu[ag]*(icu_days + mv_days)
        totalqaly = totalqaly / 365

        split_data[ag+1, 1] += totalqaly
    end
    split_data[1, 1] = sum(split_data[2:end, 1])  # sum up the qalys for each age group for the top row 
    
    all_dead = findall(x -> x.rsvtype == DEATH, humans) 
    wgt_qalylost = [9.47, 7.79, 5.93, 4.49, 2.97, 1.49]
    for i in all_dead
        x = humans[i] 
        ag = x.agegroup
        qalylost = wgt_qalylost[ag]
        split_data[ag+1, 2] += qalylost
    end
    split_data[1, 2] = sum(split_data[2:end, 2])  # sum up the qalys for each age group for the top row

    return split_data
end
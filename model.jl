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
    vaxeff_op::Vector{Float64} = zeros(Float64, 9)
    vaxeff_ip::Vector{Float64} = zeros(Float64, 9)
end
Base.show(io::IO, ::MIME"text/plain", z::Human) = dump(z)

## system parameters
Base.@kwdef mutable struct ModelParameters    ## use @with_kw from Parameters
    popsize::Int64 = 100000
    numofsims::Int64 = 1000
    usapopulation = 78_913_275
    vaccine_scenario::VAXTYPE = GSK # 0 - gsk, 1 pfizer 
    vaccine_coverage::Float64 = 1.0 
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
    p.vaccine_scenario = GSK
    p.vaccine_coverage = 1.0
    fname = string(p.vaccine_scenario)
    fname_coverage = "$(string(p.vaccine_coverage * 100))" # edit these if changing the parameters

    names = ["simid", "scenario", "num_gsk", "num_pfi", "total_sick", "nma", "outpatients", "ed", "totalhosp", "gw", "icu", "mv", "deaths", "nmadays", "sympdays", "hospdays",  "icu_nmv", "icu_mv", "totalqalys", "qalyslost"]
    df_novax = DataFrame([name => [] for name in names])
    df_wivax = DataFrame([name => [] for name in names])
    
    for i in ProgressBar(1:100)
        #@info "simulation $i"
        initialize() # will return the number of vaccines 
        
        # ORDER OF FUNCTIONS IS IMPORTANT HERE! 

        # incidence without vaccine
        incidence()
        
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
    # gsk_outpatient = [0.866, 0.853, 0.839, 0.826, 0.813, 0.800, 0.786, 0.773, 0.760]
    # gsk_inpatient = [0.998, 0.979, 0.960, 0.941, 0.922, 0.903, 0.884, 0.865, 0.846]
    # pfi_outpatient = [0.732, 0.705, 0.678, 0.651, 0.624, 0.597, 0.570, 0.543, 0.516]
    # pfi_inpatient = [0.941, 0.923, 0.906, 0.889, 0.872, 0.855, 0.838, 0.820, 0.803]
    gsk_outpatient = [0.82, 0.82, 0.82, 0.82, 0.82, 0.82, 0.81, 0.80, 0.79]
    gsk_inpatient = [0.94, 0.93, 0.93, 0.93, 0.92, 0.91, 0.90, 0.89, 0.87]
    pfi_outpatient = [0.65, 0.65, 0.64, 0.64, 0.63, 0.62, 0.61, 0.60, 0.58]
    pfi_inpatient = [0.89, 0.89, 0.88, 0.88, 0.88, 0.88, 0.87, 0.86, 0.85]

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

function _incidence_hospital(hosp_by_month::AbstractArray, comorbid_category) 
    hosp_age_distr = [0.062, 0.126, 0.265, 0.548]

    # repeat the same for hospitalization (for agents with no comorbidity)
    total_hosp = sum(hosp_by_month)  # total hospitalization (found by summing over the 9 months)
    hosp_by_age = round.(Int, hosp_age_distr .* total_hosp) # split the total hospitalization into 4 agegroups
    hosp_by_age[1] = hosp_by_age[1] + (total_hosp - sum(hosp_by_age)) # if off by 1, add the difference to the first age group 
    @assert sum(hosp_by_age) == total_hosp # sanity check
    
    percent_icu_cormobid = [0.24, 0.15, 0.12]
    picu = percent_icu_cormobid[comorbid_category+1] # since arg is 0 based
    total_icu = Int(round(total_hosp * picu))

    h_ag1 = findall(x -> x.agegroup == 1 && x.rsvtype == SUSC && x.comorbidity == comorbid_category, humans)[1:hosp_by_age[1]]
    h_ag2 = findall(x -> x.agegroup in (2, 3) && x.rsvtype == SUSC && x.comorbidity == comorbid_category, humans)[1:hosp_by_age[2]]
    h_ag3 = findall(x -> x.agegroup in (4, 5) && x.rsvtype == SUSC && x.comorbidity == comorbid_category, humans)[1:hosp_by_age[3]]
    h_ag4 = findall(x -> x.agegroup == 6 && x.rsvtype == SUSC && x.comorbidity == comorbid_category, humans)[1:hosp_by_age[4]]
    h_ag = [h_ag1..., h_ag2..., h_ag3..., h_ag4...]
    
    hospital_months_c2 = inverse_rle([1, 2, 3, 4, 5, 6, 7, 8, 9],  hosp_by_month)

    for x in h_ag
        humans[x].rsvmonth = pop!(hospital_months_c2)
        humans[x].rsvtype = HOSP
        if total_icu > 0 
            humans[x].rsvtype = ICU # overwrite with ICU
            total_icu -= 1
        end
    end

end

function incidence() 
    # sample the annual incidence 
    inc_outpatient = rand(Uniform(1595, 2669))
    inc_emergency = rand(Uniform(23, 387)) 
    inc_hospital = rand(Uniform(178, 250)) * transpose([0.055, 0.782, 0.163]) # split hospitalization over comorbid peopl
    incidence = [inc_outpatient, inc_emergency, inc_hospital...]
    @info "sampled incidence" incidence
   
    # split over 9 months, creates a matrix with 5 columns 
    # column 1: outpatients over 9 months (september, october, november, december, january, february, march, april, may)
    # column 2: emergency over 9 months 
    # column 3-5: hospitalization over 9 months (split over comorbidity)
    incidence_per_month = Int.(round.([0.0017, 0.0165, 0.0449, 0.1649, 0.3047, 0.2365, 0.1660, 0.0591, 0.0057] .* transpose(incidence)))
    @info "total incidence and (op, ed, hosp(0comorbid), hosp(2-4co), hosp(4+co)): " sum(incidence_per_month) sum(incidence_per_month, dims=1) 

    # distribute outpatients to the population
    outpatient_months = inverse_rle([1, 2, 3, 4, 5, 6, 7, 8, 9],  incidence_per_month[:, 1])
    total_outpatients = length(outpatient_months)
    non_sick_humans = humans[findall(x -> x.rsvmonth == 0, humans)[1:total_outpatients]] # don't need to sample here since the `initialize` function already shuffled the population
    for i in eachindex(outpatient_months)
        non_sick_humans[i].rsvmonth = outpatient_months[i] 
        non_sick_humans[i].rsvtype = OP
    end

    # repeat the same for emergency
    emergency_months = inverse_rle([1, 2, 3, 4, 5, 6, 7, 8, 9],  incidence_per_month[:, 2])
    total_emergencies = length(emergency_months)
    non_sick_humans = humans[findall(x -> x.rsvmonth == 0, humans)[1:total_emergencies]] # don't need to sample here since the `initialize` function already shuffled the population
    for i in eachindex(emergency_months)
        non_sick_humans[i].rsvmonth = emergency_months[i] 
        non_sick_humans[i].rsvtype = ED
    end

    #  hospitalization
    _incidence_hospital(incidence_per_month[:, 3], 0) # 0 no comorbidity
    _incidence_hospital(incidence_per_month[:, 4], 1) # 1: 1 - 3 comorbidities
    _incidence_hospital(incidence_per_month[:, 5], 2) # 
  
    # distribute mechanical ventilation to ICU admissions 
    mv_elg = shuffle(findall(x -> x.rsvtype == ICU, humans))
    mv_tot = Int(round(length(mv_elg) * 0.166)) 
    for x in humans[mv_elg[1:mv_tot]] 
        x.rsvtype = MV
    end 

    # distribute deaths among hospitalized indivbiduals
    _all_hosp = length(findall(x -> x.rsvtype in (HOSP, ICU, MV), humans)) 
    all_hosp = _all_hosp * rand(Uniform(0.056, 0.076)) # 5.6% to 7.6% of all hospitalized patients die
    cnt_death_nonicu, cnt_death_icu = Int.(round.(all_hosp .* [0.37, 0.63])) # hosp rate for ICU and non ICU patients
    idx_death_nonicu = shuffle(findall(x -> x.rsvtype == HOSP, humans))[1:cnt_death_nonicu]
    idx_death_icu = shuffle(findall(x -> x.rsvtype in (ICU, MV), humans))[1:cnt_death_icu]
    for i in [idx_death_nonicu..., idx_death_icu...]
        humans[i].rsvtype = DEATH
    end

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
        hospdays += 2  # since person did not die, 2 days of recovery
    elseif rsvtype == ICU || rsvtype == MV 
        sympdays += 4  # 4 days of symptoms before hospital admission
        hospdays += 1  # one day general ward before ICU 
        if rsvtype == ICU 
            icu_nmv_days += rand(Gamma(1.5625,2.8802))
        else 
            icu_mv_days += rand(Gamma(1.5625,2.8802)) 
        end
        hospdays += 2  # since person did not die 
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
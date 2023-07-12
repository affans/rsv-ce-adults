using StatsBase, Random, Distributions, DataFrames, Logging

@enum INFTYPE SUSC=0 NMA=1 OP=2 ED=3 HOSP=4
@enum VAXTYPE GSK=1 PFI=2
@enum VAXMONTH SEP=1 OCT=2

# define an agent and all agent properties
Base.@kwdef mutable struct Human
    idx::Int64 = 0 
    agegroup::Int64 = 0    # in years. don't really need this but left it incase needed later
    comorbidity::Int64 = 0 # 0, 1=(1 - 3 comorbidity), 2=(4+ comorbidity)
    rsvtype::INFTYPE = SUSC # 1 = outpatient, 2 = emergency, 3 = hospital
    rsvmonth::Int64 = 0  # from months 1 to 9 
    rsvicu::Bool = false # 1 = yes, 0 = no
    rsvmv::Bool = false # 1 = yes, 0 = no
    rsvdeath::Bool = false 
    rsvsympdays::Int64 = 0 
    rsvicudays::Int64 = 0 
    vaccinated::Bool = false 
    vaxmonth::VAXMONTH = SEP  # defauly 
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
    vaccine_scenario::Float64 = 0 # 0 - gsk, 1 pfizer 
end

# constant variables
const humans = Array{Human}(undef, 0) 
const p = ModelParameters()  ## setup default parameters
pc(x) = Int(round(x / p.usapopulation * p.popsize)) # convert to per-capita

function simulations()
    # redirect the @info statements
    logger = NullLogger()
    global_logger(logger)

    reset_params() 
    Random.seed!(53)

    names = ["simid", "scenario", "total_sick", "nma", "outpatients", "ed", "hosp", "icu", "mv", "sympdays", "hospdays"]
    df = DataFrame([name => [] for name in names])
    for i = 1:1000
        @info "simulation $i"
        initialize()
        incidence()
        all_sick, nma, outpatients, emergency, hosp, totalicu, totalmv = collect_incidence() 
        symp, hosp = calculate_days() 
        push!(df, [i, "novax", all_sick, nma, outpatients, emergency, hosp, totalicu, totalmv, symp, hosp])
        vaccine() 
        all_sick, nma, outpatients, emergency, hosp, totalicu, totalmv = collect_incidence() 
        symp, hosp = calculate_days() 
        push!(df, [i, "wivax", all_sick, nma, outpatients, emergency, hosp, totalicu, totalmv, symp, hosp])
    end
    df
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

    gsk_outpatient = [0.866, 0.853, 0.839, 0.826, 0.813, 0.800, 0.786, 0.773, 0.760]
    gsk_inpatient = [0.998, 0.979, 0.960, 0.941, 0.922, 0.903, 0.884, 0.865, 0.846]
    pfi_outpatient = [0.732, 0.705, 0.678, 0.651, 0.624, 0.597, 0.570, 0.543, 0.516]
    pfi_inpatient = [0.941, 0.923, 0.906, 0.889, 0.872, 0.855, 0.838, 0.820, 0.803]

    # 90% of individuals will get vaccine
    # 50% in september, 50% in october 

    # september vaccinations
    for i = 1:45000
        x = humans[i]
        x.vaccinated = true
        x.vaxmonth = SEP

        vax_select = rand() < p.vaccine_scenario ? :pfi : :gsk
        if vax_select == :gsk
            x.vaxeff_op = roll_vaccine(1, gsk_outpatient)
            x.vaxeff_ip = roll_vaccine(1, gsk_inpatient)
        elseif vax_select == :pfi
            x.vaxeff_op = roll_vaccine(1, pfi_outpatient)
            x.vaxeff_ip = roll_vaccine(1, pfi_inpatient)
        end
    end        

    # october vaccinations
    for i = 45001:90000
        x = humans[i]
        x.vaccinated = true
        x.vaxmonth = OCT 
        vax_select = rand() < p.vaccine_scenario ? :pfi : :gsk
        if vax_select == :gsk
            x.vaxeff_op = roll_vaccine(2, gsk_outpatient)
            x.vaxeff_ip = roll_vaccine(2, gsk_inpatient)
        elseif vax_select == :pfi
            x.vaxeff_op = roll_vaccine(2, pfi_outpatient)
            x.vaxeff_ip = roll_vaccine(2, pfi_inpatient)
        end
    end
        
    shuffle!(humans) # shuffle the humans for stochastictiy
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
    @info "total incidence: " sum(incidence_per_month) sum(incidence_per_month, dims=1) 

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


    # repeat the same for hospitalization (for agents with no comorbidity)
    hospital_months_c1 = inverse_rle([1, 2, 3, 4, 5, 6, 7, 8, 9],  incidence_per_month[:, 3])
    total_hospitals = length(hospital_months_c1)
    total_icu = Int(round(total_hospitals * 0.24))
    non_sick_humans = humans[findall(x -> x.rsvmonth == 0 && x.comorbidity == 0, humans)[1:total_hospitals]] # don't need to sample here since the `initialize` function already shuffled the population
    for i in eachindex(hospital_months_c1)
        non_sick_humans[i].rsvmonth = hospital_months_c1[i] 
        non_sick_humans[i].rsvtype = HOSP
        if total_icu > 0
            non_sick_humans[i].rsvicu = true
            total_icu -= 1
        end
    end

    # repeat the same for hospitalization (for agents with 1-3 comorbidity)
    hospital_months_c2 = inverse_rle([1, 2, 3, 4, 5, 6, 7, 8, 9],  incidence_per_month[:, 4])
    total_hospitals = length(hospital_months_c2)
    total_icu = Int(round(total_hospitals * 0.15))
    non_sick_humans = humans[findall(x -> x.rsvmonth == 0 && x.comorbidity == 1, humans)[1:total_hospitals]] # don't need to sample here since the `initialize` function already shuffled the population
    for i in eachindex(hospital_months_c2)
        non_sick_humans[i].rsvmonth = hospital_months_c2[i] 
        non_sick_humans[i].rsvtype = HOSP
        if total_icu > 0 
            non_sick_humans[i].rsvicu = true 
            total_icu -= 1
        end
    end

    # repeat the same for hospitalization (for agents with 4+ comorbidity)
    hospital_months_c3 = inverse_rle([1, 2, 3, 4, 5, 6, 7, 8, 9],  incidence_per_month[:, 5])
    total_hospitals = length(hospital_months_c3)
    total_icu = Int(round(total_hospitals * 0.12))
    non_sick_humans = humans[findall(x -> x.rsvmonth == 0 && x.comorbidity == 2, humans)[1:total_hospitals]] # don't need to sample here since the `initialize` function already shuffled the population
    for i in eachindex(hospital_months_c3)
        non_sick_humans[i].rsvmonth = hospital_months_c3[i] 
        non_sick_humans[i].rsvtype = HOSP
        if total_icu > 0 
            non_sick_humans[i].rsvicu = true 
            total_icu -= 1
        end
    end

    # distribute mechanical ventilation to ICU admissions 
    mv_elg = findall(x -> x.rsvicu == true, humans) 
    mv_tot = Int(round(length(mv_elg) * 0.166)) 
    for x in humans[mv_elg[1:mv_tot]] 
        x.rsvmv = true
    end 

    # distribute deaths
    _all_hosp = length(findall(x -> x.rsvtype == HOSP, humans)) 
    all_hosp = _all_hosp * rand(Uniform(0.056, 0.076))
    cnt_death_nonicu, cnt_death_icu = Int.(round.(all_hosp .* [0.37, 0.63])) # hosp rate for ICU and non ICU patients
    idx_death_nonicu = findall(x -> x.rsvtype == HOSP && x.rsvicu == false, humans)[1:cnt_death_nonicu]
    idx_death_icu = findall(x -> x.rsvtype == HOSP && x.rsvicu == true, humans)[1:cnt_death_icu]
    for i in [idx_death_nonicu..., idx_death_icu...]
        humans[i].rsvdeath = true
    end

end

function calculate_inf_days(x::Human) 
    sympdays = 0 
    hospdays = 0 
    if x.rsvtype == NMA 
        sympdays += rand(Uniform(2, 8)) 
    elseif x.rsvtype == OP 
        sympdays += rand(Uniform(7, 14))
    elseif x.rsvtype == ED
        sympdays += rand(Uniform(7, 14))
    elseif x.rsvtype == HOSP
        hospdays = 4 
        if x.rsvicu  
            hospdays += rand(Gamma(1.5625,2.8802))
        else 
            hospdays += rand(Gamma(1.2258, 5.0582))
        end 
        if !x.rsvdeath 
            hospdays += 2 
        end
    end
    return (sympdays, hospdays)
end

function calculate_days() 
    total_symp_days = 0 
    total_hosp_days = 0 
    all_sick = findall(x -> Int(x.rsvtype) > 0, humans)
    for i in all_sick
        s, h = calculate_inf_days(humans[i])
        total_symp_days += s 
        total_hosp_days += h
    end
    return (total_symp_days, total_hosp_days)
end

function collect_incidence()
    # check 
    all_sick = length(findall(x -> Int(x.rsvtype) > 0, humans))  
    nma = length(findall(x -> x.rsvtype == NMA, humans))
    outpatients = length(findall(x -> x.rsvtype == OP, humans))   
    emergency = length(findall(x -> x.rsvtype == ED, humans))   
    hosp = length(findall(x -> x.rsvtype == HOSP, humans))
    totalicu = length(findall(x -> x.rsvicu == true, humans))
    totalmv = length(findall(x -> x.rsvmv == true, humans))

    return (all_sick, nma, outpatients, emergency, hosp, totalicu, totalmv)
    
    # for statistics only
    # hosp_no_comorbid  = length(findall(x -> x.rsvtype == HOSP && x.comorbidity == 0, humans))
    # hosp_13_comorbid = length(findall(x -> x.rsvtype == HOSP && x.comorbidity == 1, humans))
    # hosp_4_comorbid = length(findall(x -> x.rsvtype == HOSP && x.comorbidity == 2, humans))
    #icu_no_comorbid  = length(findall(x -> x.rsvtype == 3 && x.comorbidity == 0 && x.rsvicu == true, humans))
    #icu_13_comorbid = length(findall(x -> x.rsvtype == 3 && x.comorbidity == 1 && x.rsvicu == true, humans))
    #icu_4_comorbid = length(findall(x -> x.rsvtype == 3 && x.comorbidity == 2 && x.rsvicu == true, humans))
    #@info "total humans sick " all_sick outpatients emergency hosp hosp_no_comorbid  hosp_13_comorbid hosp_4_comorbid icu_no_comorbid #icu_13_comorbid icu_4_comorbid
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
                x.rsvicu = false
                x.rsvmv = false
                x.rsvdeath = false
                op_to_nma += 1
            end
        end 
        if x.rsvtype == HOSP 
            if rn < x.vaxeff_ip[x.rsvmonth]
                x.rsvtype = OP
                x.rsvicu = false
                x.rsvmv = false
                x.rsvdeath = false
                ip_to_op += 1
            end
        end 
    end
    return (op_to_nma, ip_to_op)
end


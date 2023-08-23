
#----Simulation: functions----
validSimulationGRN <- function(object) {
	#graph is a GraphGRN object
	if (!is(object@graph, 'GraphGRN')) {
		stop('graph must be a GraphGRN object')
	}
	
	#local noise
	if (object@expnoise<0) {
		stop('Ecperimental noise standard deviation must be greater than 0')
	}
	
	#global noise
	if (object@bionoise<0) {
		stop('Biological noise standard deviation must be greater than 0')
	}
	
	return(TRUE)
}

initSimulationGRN <- function(.Object, ..., graph, kd_genes, propkd=0.5, expnoise = 0, bionoise = 0, seed = sample.int(1e6,1), inputModels = list(), propBimodal = 0) {
	.Object@graph = graph
	.Object@expnoise = expnoise
	.Object@bionoise = bionoise
	.Object@seed = seed
	.Object@inputModels = inputModels

	if (length(inputModels) == 0) {
	  message("running generateInputModels because inputModels is of length 0")
	  .Object = generateInputModelsModified(.Object, kd_genes, propkd)
	}
	
	validObject(.Object)
	return(.Object)
}

solveSteadyState <- function(object, externalInputs) {
  
  #external inputs
  if (is.null(names(externalInputs)) |
      !all(names(externalInputs) %in% getInputNodes(object@graph))) {
    stop('Invalid external inputs vector, named vector expected for ALL input nodes')
  }
  
  #set random seed
  set.seed(object@seed)
  
  #solve ODE
	ode = generateODE(object@graph)
	ext = externalInputs
	graph = object@graph
	nodes = setdiff(nodenames(graph), names(ext))
	exprs = rbeta(length(nodes), 2, 2)
	exprs[exprs < 0] = 0
	exprs[exprs > 1] = 1
	names(exprs) = nodes
	
	soln = nleqslv(exprs, ode, jac = NULL, ext)
	
	#check if convergence is reached or not
	if(soln$termcd != 1) {
	  warning('Solution not achieved. use \'diagnostics(simulation)\' to get details')
	}
	return(soln)
}

# ----------------Modified version-----------------
createInputModelsModified <- function(simulation, bimodal_genes, propkd = 0.5) {
  
  # simulation is a SimulationGRN object
  
  set.seed(simulation@seed)
  
  #create input models
  innodes = getInputNodes(simulation@graph)
   
  inmodels = list()
  
  # For each input node, determine parameters
  # for each knockdown gene, expression profiles with the knockdown 
  # followed a binomial distribution Binomial(p=p,N=500) 
    
  # Modified
  for (n in innodes) {
    
    parms = list()
        
    # Modified
    if (n %in% bimodal_genes) {
      
      print(paste("Knocked down:", n))
      
      # If it is knockdown, sample from binomial distribution
      # parms = c(parms, 'prop' = runif(1, 0.2, 0.8))
      parms = c(parms, 'prop' = propkd)
      # Proportions of wt and kd
      parms$prop = c(parms$prop, 1 - parms$prop)
      # normal distribution means were sampled from a B(10,10) beta distribution
      # for wildtype genes or B(10,100) for knocked down genes. 
      parms$mean = c(rbeta(1, 10, 100), rbeta(1, 10, 10))
      
      # print(parms)
      
    } else { # No knockdown
      
      # print(paste("not knocked down:", n))
      parms$prop = 1
      parms$mean = rbeta(1, 10, 10)
      # print(parms)
    }
    
    # The normal distribution variances were sampled from B(15,15)
    # and then scaled by min(u,(1-u))/3 where u is the mean; 
    # scaling ensured that support for the normal distributions was
    # concentrated within the range [0,1]. 
    maxsd = pmin(parms$mean, 1 - parms$mean) / 3
    parms$sd = sapply(maxsd, function(x) max(rbeta(1, 15, 15) * x, 0.01))
    inmodels = c(inmodels, list(parms)) # input models
  }
  
  names(inmodels) = innodes
  simulation@inputModels = inmodels
  
  return(simulation)
}


#src = https://stats.stackexchange.com/questions/2746/how-to-efficiently-generate-random-positive-semidefinite-correlation-matrices
#src = Lewandowski, Kurowicka, and Joe (LKJ), 2009
#lower betaparam gives higher correlations
vineS <- function(d, betaparam = 5, seed = sample.int(1E6, 1)) {
  
  set.seed(seed)
  P = matrix(rep(0, d ^ 2), ncol = d) 
  S = diag(rep(1, d)) 
  
  for (k in 2:(d - 1)) {
    for (i in (k + 1):d) {
      
      P[k, i] = rbeta(1, betaparam, betaparam)
      P[k, i] = (P[k, i] - 0.5) * 2
      p = P[k, i]
      for (l in (k - 1):1) {
        p = p * sqrt((1 - P[l, i] ^ 2) * (1 - P[l, k] ^ 2)) + P[l, i] * P[l, k]
      }
      S[k, i] = p
      S[i, k] = p
    }
  }
  
  permutation = sample(1:d, d) # no replacement, just permute
  S = S[permutation, permutation]
  
  return(S)
}

generateInputDataModified <- function(simulation, bimodal_genes, numsamples, cor.strength = 5, propkd = 0.5) {
  
  # Function to simulate starting expression values of input nodes
  # Considering correlated inputs based on beta parameter
  
  set.seed(simulation@seed)
  
  innodes = getInputNodes(simulation@graph)
  
  # Initialize matrix with -1
  # Rows: samples, columns: inputnodes
  externalInputs = matrix(-1,nrow = numsamples, ncol = length(innodes))
  colnames(externalInputs) = innodes
    
  #create input models
  if (length(simulation@inputModels) == 0) {
    simulation = generateInputModelsModified(simulation, bimodal_genes, propkd)
  }
  
  #simulate external inputs
  inmodels = simulation@inputModels
  classf = c()
  
  # for each input node (column), fill out starting expression values
  for (n in innodes) { 
    
    m = inmodels[[n]] # prop, mean, sd
    
    if (length(m$prop) == 2){
      wts = rep(c(1),(numsamples * m$prop[1]))
      kds = rep(c(2),(numsamples * m$prop[2]))
      mix = c(wts, kds)
    } else{
      mix = sample(1:length(m$prop), numsamples, prob = m$prop, replace = T)
    }

    outbounds = 1
    while (sum(outbounds) > 0){
      
      # outbounds is vector of logical
      outbounds = externalInputs[ , n] < 0 | externalInputs[ , n] > 1
      # m -> stats for each input node
      # mix = 1 if not a bimodal gene
      externalInputs[outbounds & mix == 1, n] = rnorm(sum(outbounds & mix == 1), m$mean[1], m$sd[1])
      if (length(m$prop) > 1) { # bimodal gene, (mix = 2), sample from 
        externalInputs[outbounds & mix == 2, n] = rnorm(sum(outbounds & mix == 2), m$mean[2], m$sd[2])
      }
    }
    
    # save infor only for bimodal gene --> knockdown
    if (length(m$prop) > 1) {
      #save class information
      #current input node n
      classf = rbind(classf, mix)
      rownames(classf)[nrow(classf)] = n
    }
  }
  
  #correlated inputs
  if (cor.strength > 0 & numsamples > 1) {
    
    inputs = ncol(externalInputs)
    # Sort values in each column in increasing order
    dm = apply(externalInputs, 2, sort)
    # Covariance matrix
    covmat = vineS(inputs, cor.strength, simulation@seed)

    # Produce samples from covmat
    # n: no. of samples required
    # mu = rep(0, inputs) = vector of means
    # sample from the specified multivariate normal distribution
    cordata = mvrnorm(numsamples, rep(0, inputs), covmat)
    
    for (i in 1:inputs) {
      #avoid correlated bimodal inputs
      # if input is bimodal
      # The current input node should be independent of other
      if (i %in% which(innodes %in% rownames(classf))) {
        cordata[, i] = externalInputs[, i]
      } else {
        # rank the values
        cordata[, i] = dm[, i][rank(cordata[, i])]
      }
    }
    
    externalInputs = cordata
  }
  
  #add mixture info to attributes
  attr(externalInputs, 'classf') = classf
  
  colnames(externalInputs) = innodes
  return(externalInputs)
}

# A main function
#cor.strength used for generating correlated inputs
simDataset <- function(simulation, bimodal_genes, numsamples, cor.strength, externalInputs) {
  
  if (missing(cor.strength)) {
    cor.strength = 5
  }
    
  #generate input matrix
  innodes = getInputNodes(simulation@graph)
  class(innodes[1])
  
  message("innodes", innodes)
  
  if (!missing(externalInputs) && !is.null(externalInputs)) {
    print("missing externalinputs")
    if (nrow(externalInputs) != numsamples |
        length(setdiff(innodes, colnames(externalInputs))) != 0) {
          stop('Invalid externalInputs matrix provided')
    }
    externalInputs = externalInputs[, innodes, drop = F]
    classf = NULL
  } else{
    
    # source of NA/Inf replaced by maximum positive value
    externalInputs = generateInputDataModified(simulation, bimodal_genes, numsamples, cor.strength)
    
    #extract class information
    classf = attr(externalInputs, 'classf')
  }
  
  #set random seed
  set.seed(simulation@seed)

  #solve ODE
  graph = simulation@graph
  ode = generateODE(graph)
  
  #generate LN noise for simulation
  lnnoise = exp(rnorm(numsamples * length(nodenames(graph)), 0, simulation@bionoise))
  lnnoise = matrix(lnnoise, nrow = numsamples, byrow = T)
  colnames(lnnoise) = nodenames(graph)
  
  #initialize solutions
  nodes = setdiff(nodenames(graph), colnames(externalInputs))
  exprs = rbeta(length(nodes) * numsamples, 2, 2)
  exprs[exprs < 0] = 0
  exprs[exprs > 1] = 1
  exprs = matrix(exprs, nrow = numsamples)
  colnames(exprs) = nodes

  #solve ODEs for different inputs
  res = foreach(i = 1:numsamples, .packages = c('nleqslv'), .combine = cbind) %dopar% {
    soln = nleqslv(exprs[i, ], ode, externalInputs = externalInputs[i, ], lnnoise = lnnoise[i, ])
    return(c(soln$x, soln$termcd))
  }
  
  message("finished foreach")
  
  res = cbind(res)
  
  termcd = res[nrow(res),] # error codes fro nleqslv
  emat = res[-(nrow(res)), , drop = F]
  emat = rbind(emat, t(externalInputs))
  colnames(emat) = paste0('sample_', 1:numsamples)

  #check for errors
  if (!all(termcd == 1)) {
    nc = termcd != 1
    msg = 'Simulations for the following samples did not converge:'
    sampleids = paste(colnames(emat)[nc], ' (', termcd[nc], ')', sep = '')
    msg = paste(c(msg, sampleids), collapse = '\n\t')
    msg = paste(msg, 'format: sampleid (termination condition)', sep = '\n\n\t')
    warning(msg)

    emat = emat[, !nc]
  }
  
  #add experimental noise
  expnoise = rnorm(nrow(emat) * ncol(emat), 0, simulation@expnoise)
  expnoise = matrix(expnoise, nrow = nrow(emat), byrow = T)
  emat = emat + expnoise
  
  #add class information to attributes
  if (!is.null(classf)) {
    classf = classf[, termcd == 1, drop = F]
    colnames(classf) = colnames(emat)
    attr(emat, 'classf') = classf
  }
  
  return(emat)
}

generateSensMat <- function(simulation, pertb, inputs = NULL, pertbNodes = NULL, tol = 1E-3) {
  set.seed(simulation@seed)
  graph = simulation@graph
  
  if (pertb < 0 | pertb > 1) {
    stop('Perturbation (knock-down) should be between 0 and 1.')
  }
  
  if (is.null(inputs)) {
    inputs = runif(length(getInputNodes(graph)), pertb + 1E-4, 1)
    names(inputs) = getInputNodes(graph)
  }else if (!all(getInputNodes(graph) %in% names(inputs))) {
    stop('Missing Inputs')
  }
  
  if (is.null(pertbNodes)) {
    pertbNodes = nodenames(graph)
  } else{
    pertbNodes = intersect(pertbNodes, nodenames(graph))
  }
  
  #generate ODE functions
  graph = simulation@graph
  ode = generateODE(graph)
  
  #generate LN noise for simulation - 0 noise
  lnnoise = exp(rep(0, length(nodenames(graph))))
  names(lnnoise) = nodenames(graph)
  
  #initialize solutions
  nodes = setdiff(nodenames(graph), names(inputs))
  exprs = rbeta(length(nodes) * (1 + length(pertbNodes)), 2, 2)
  exprs[exprs < 0] = 0
  exprs[exprs > 1] = 1
  exprs = matrix(exprs, nrow = (1 + length(pertbNodes)))
  colnames(exprs) = nodes
  
  #original solution with no perturbations
  soln0 = nleqslv(exprs[(1 + length(pertbNodes)), ], ode, externalInputs = inputs, lnnoise = lnnoise)
  termcd0 = soln0$termcd
  soln0 = soln0$x
  soln0 = c(inputs, soln0)
  
  #solve ODEs for different perturbations
  res = foreach(i = 1:length(pertbNodes), .combine = rbind) %do% {
    n = pertbNodes[i]
    tempinputs = inputs
    if (n %in% names(inputs)){
      odefn = ode
      tempinputs[n] = max(tempinputs[n] * (1 - pertb), 1E-4)
    } else{
      getNode(graph, n)$spmax = getNode(graph, n)$spmax - pertb
      odefn = getODEFunc(graph)
      getNode(graph, n)$spmax = getNode(graph, n)$spmax + pertb
    }
    
    soln = nleqslv(exprs[i, ], odefn, externalInputs = tempinputs, lnnoise = lnnoise)
    tcd = soln$termcd
    soln = soln$x
    soln = c(tempinputs, soln, tcd)
    return(soln)
  }
  res = rbind(c(), res) #if numeric vector returned, convert to matrix
  termcds = res[, ncol(res), drop = F]
  res = res[, -ncol(res), drop = F]
  res = rbind(res)
  
  sensmat = c()
  for (i in 1:length(pertbNodes)) {
    n = pertbNodes[i]
    
    #calculate sensitivity
    diffexpr = (res[i, ] - soln0)
    diffexpr[abs(diffexpr) < tol] = 0 #small difference resulting from numerical inaccuracies
    sensmat = rbind(sensmat, diffexpr/-pertb * getNode(graph, n)$spmax/soln0)
  }
  
  #if base case does not converge, no sensitivity analysis possible
  if (termcd0 != 1) {
    warning('Convergance not achieved for unperturbed case')
    sensmat = sensmat * 0
  }
  
  #if some solutions do not converge, set all their sensitivities to 0
  termcds = termcds == 1
  if (!all(termcds)) {
    warning('Convergance not achieved for SOME perturbations')
    termcds = as.numeric(termcds)
    termcds = termcds %*% t(rep(1, ncol(sensmat)))
    sensmat = sensmat * termcds
  }
  
  rownames(sensmat) = pertbNodes
  
  #since sensitivity calculation depends on the solver, round sensitivities to 
  #account for such numerical inaccuracies
  sensmat = round(sensmat, digits = round(-log10(tol)))
  
  return(sensmat)
}

#only derives the truth when the bimodal genes are input nodes
getGoldStandardModified <- function(simulation, sensmat = NULL) {

  #extract variables from the model
  graph = simulation@graph
  
  #get bimodal genes
  bimodal = unlist(lapply(simulation$inputModels, function(x) length(x$prop)))
  bimodal = names(bimodal)[bimodal == 2]
  if (length(bimodal) == 0) {
    stop('No conditional associations in the network')
  }
  
  #perform sensitivity analysis on the model if required
  inputs = sapply(simulation$inputModels, function(x) x$mean[1])
  inputs[bimodal] = 0.5
  names(inputs) = getInputNodes(graph)
  if (is.null(sensmat)) {
    sensmat = sensitivityAnalysis(simulation, 0.25, inputs, nodenames(graph))
  } else if (!all(rownames(sensmat) %in% names(inputs)) |
             !all(colnames(sensmat) %in% nodenames(graph))) {
    stop('Sensitivity matrix must be square and with all genes perturbed')
  }
  
  sensmat[cbind(rownames(sensmat), rownames(sensmat))] = 0
  sensmat = abs(sensmat) > 0.01
  
  #generate condcoex mat
  condcoexmat = sensmat[bimodal, , drop = F] * 0
  triplets = c()
  for (b in bimodal) {
    if ((sum(sensmat[bimodal,]) == 0) | is.na(sum(sensmat[bimodal,]))){
      next
	}
    
    #identify direct targets and conditionally regulated targets
    bmat = sensmat[, sensmat[b, ], drop = F]
    bmat = bmat[rowSums(bmat) != 0, , drop = F]
    condcoex = colnames(bmat)[bmat[b, ] & colSums(bmat) == 1]
    coregtgts = colnames(bmat)[bmat[b, ] & colSums(bmat) > 1]
    condcoexmat[b, condcoex] = 1
    
    #identify conditionally dependent pairs
    bmat = bmat[!rownames(bmat) %in% b, coregtgts, drop = F]
    diffpairs = melt(bmat)
    diffpairs = diffpairs[diffpairs$value, 1:2]
    colnames(diffpairs) = c('TF', 'Target')
    diffpairs$TF = as.character(diffpairs$TF)
    diffpairs$Target = as.character(diffpairs$Target)
    if (nrow(diffpairs) == 0)
      next
    
    #select downstream genes for coregulating inputs
    diffpairs = ddply(diffpairs, 'Target', function(x) {
      newtfs = colnames(sensmat)[colSums(sensmat[x$TF, , drop = F]) == colSums(sensmat) &
                                   colSums(sensmat) != 0]
      diffdf = data.frame('TF' = c(x$TF, newtfs), stringsAsFactors = F)
      return(diffdf)
    })
    
    diffpairs = diffpairs[diffpairs$TF != diffpairs$Target, ]
    triplets = rbind(triplets, cbind('cond' = b, diffpairs[, 2:1]))
  }
  
  #restructure triplets dataframe
  rownames(triplets) = NULL
  triplets$known = T
  
  #distances
  nodedist = distances(GraphGRN2igraph(graph), mode = 'out')
  nodedist = melt(nodedist)
  names(nodedist) = c('TF', 'Target', 'Dist')
  triplets = merge(triplets, nodedist, all.x = T)
  triplets = triplets[, c(3, 1:2, 4:ncol(triplets))]
  triplets$Direct = triplets$Dist==1
  triplets$Influence = !is.infinite(triplets$Dist)
  triplets$Association = T
  
  #export condcoexmat as attribute
  attr(triplets, 'condcoex') = condcoexmat
  
  return(triplets)
}

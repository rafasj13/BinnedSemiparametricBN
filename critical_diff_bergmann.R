install.packages("scmamp_0.2.1.tar.gz", repos=NULL, type="source")
library(scmamp)



data <- read.csv("data/real_no_pa/slogl.csv",  header=TRUE)
data

test.res <- postHocTest(data = data, test = 'friedman', correct = 'bergmann')
test.res

average.ranking <- colMeans(rankMatrix(data, decreasing=TRUE))
average.ranking
drawAlgorithmGraph(pvalue.matrix = test.res$corrected.pval, mean.value = average.ranking)
?rankMatrix()


test.res.df <- as.data.frame(test.res$corrected.pval)
avg.ranking.df <- as.data.frame(average.ranking)
avg.ranking.df
write.csv(test.res.df, file = "result/real_no_pa/slogl_bergmann_hommel.csv", row.names = TRUE)
write.csv(avg.ranking.df, file = "result/real_no_pa/slogl_avg_ranking.csv", row.names = TRUE)



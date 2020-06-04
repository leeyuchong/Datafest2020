# An Analysis of Fall Semester Plans of U.S Universities
By Adam Chapnik; Anish Kumthekar; Yixiao (Carl) Cao; Ha Bui; Jason Lee

NOTE: The code is provided for reference and depends on the output given by sections in the other documents. 

Visualisations for the data can be found here: https://notbanner.ml/datafest2020

Specific information about this project can be found in the Wiki

# Motivation
The COVID-19 pandemic has posed multifaceted socio-economic disruptions, among which education discontinuance has threatened to both halt personal intellectual and societal progresses as well as aggravating systemic inequalities. College students are especially concerned about their institutions’ future plan for admission and operation. For many, deciding to continue their education without being able to physically interact with their instructors and peers can incur tremendous financial losses. In this study, we attempt to establish a linear model to predict the probability of reopening in the Fall semester of 2020 among higher education institutions in the U.S. using institutional characteristics such as financial resources and geographical/socio-economic backgrounds, among others, as explanatory variables. 

# Methods
Our group uses data on over 700 higher education institutions nationwide. This sample includes several indicators used to determine the appropriate response to the public health and financial threats pertaining to COVID-19. We use the schools’ plan in the fall as the dependent variable, dividing the variable into three levels: open or hybrid (1), waiting to make a decision (0.5), and online (0). This logit model allows us to yield predictions for a non-binary ordinal dependent variable. To accommodate for an ordinal outcome, we employ the multinomial logistic multiple regression. Nevertheless, in the process of working, we also notice that data clustering poses another problem. This is because schools in different states, counties, cities, etc. tended to have different financial resources alongside student-faculty and surrounding areas’ demographics, among other factors. Therefore, each state can be treated as a “group” of data that has a different residual term (assuming that inter-group residuals are independent), thus violating the assumption of independence among neighboring error terms on school (this is the lowest level of units belonging to a group) of the logistic regression model. Therefore, we also consider the multilevel model, in which we establish a random intercept (with fixed slope) and a mixed effects (with random intercept and slope) multiple linear regression model. The linear models adopted yield a continuous dependent variable - a disadvantage that the multinomial logit model could compensate. We are aware that there have been hybrids of the logistic and the multilevel models, yet have not been mathematically and technically capable of producing such a model. 

# Key Variables
- Distcrs: Whether remote courses are offered (Dummy variable)
- Applications: Number of applications per year
- Endowment: School endowment in dollars
- alloncam: Whether all students are required to stay on campus (Dummy variable)
- board: Whether room and board is provided (Categorical variable)
- number_locations: number of locations students are from (out-of-state and international)
- county: number of COVID-19 cases in the county
- saeq9at: average salary for instructional staff (specifically for associate professors) equated to a 9-month contract total
- sanin: salary outlays for full-time staff
  - sanin04: archivsts, curators, librarians/library technicians and people in academic affairs and other educational services
  - sanin05: management positions
  - sanin08: community, social services, legal, arts, design, entertainment, sports and media

# Key Findings
Our initial hypothesis of tying the sentiment score to the plans of the college, did not end up being a useful avenue, because schools that were going online, actually tended to issue more positive sentiment statements. Thus, sentiment score wasn’t a significant predictor in our final model. Similarly although multinomial logistic regression was an option to consider, when we think of the group effect, affecting every variable we needed to use the multilevel model. The significant predictors for the model were number of applications per year, endowment in $, whether students were required to stay on campus, whether board provided, number of locations the students are from, student-faculty ratio, Covid-19 cases in the county, average staff salary, salary outlays for librarians, management and social service. And finally, using the random intercept model, we get the appropriate coefficients for these predictors based on group stage: state.

# Concluding Remark
The model informed us that schools are heavily subject to states’ financial support, while students’ demographics (foreigners or native). Researchers have concurred that large and/or private schools were more likely to weather the disruption, while smaller and/or public schools whose incomes depend primarily on domestic students.

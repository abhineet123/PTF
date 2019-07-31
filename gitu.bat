git add --all .
IF "%1"=="" (
	git commit
) ELSE (
	IF "%1"=="f" (
		git submodule foreach 'git commit -m "minor fix"'
		git commit -m "minor fix"
	) ELSE (
		git submodule foreach 'git commit -m "%1"'
		git commit -m "%1"
	)
)
git push origin master
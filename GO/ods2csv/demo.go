package main

import (
	"fmt"
	"strings"
)

func ReplaceHTMLSpecialEntities1(input string) string {
	output := strings.Replace(input, "&amp;", "&", -1)
	output = strings.Replace(output, "&lt;", "<", -1)
	output = strings.Replace(output, "&gt;", ">", -1)
	output = strings.Replace(output, "&quot;", "\"", -1)
	output = strings.Replace(output, "&lsquo;", "‘", -1)
	output = strings.Replace(output, "&rsquo;", "’", -1)
	output = strings.Replace(output, "&tilde;", "~", -1)
	output = strings.Replace(output, "&ndash;", "–", -1)
	output = strings.Replace(output, "&mdash;", "—", -1)
	output = strings.Replace(output, "&apos;", "'", -1)

	return output
}

func ReplaceHTMLSpecialEntities2(input string) string {
	output := input
	replaceMap := map[string]string{
		"&amp;":   "&",
		"&lt;":    "<",
		"&gt;":    ">",
		"&quot;":  "\"",
		"&lsquo;": "‘",
		"&rsquo;": "’",
		"&tilde;": "~",
		"&ndash;": "–",
		"&mdash;": "—",
		"&apos;":  "'",
	}
	for entity, char := range replaceMap {
		output = strings.Replace(output, entity, char, -1)
	}
	return output
}

func main() {
	input := "&amp;&lt;&gt;&quot;&lsquo;&rsquo;&tilde;&ndash;&mdash;&apos;"
	fmt.Println(ReplaceHTMLSpecialEntities1(input))
	fmt.Println(ReplaceHTMLSpecialEntities2(input))
}

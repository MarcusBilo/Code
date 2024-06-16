package main

import (
	"fmt"
	"strings"
)

func ReplaceHTMLSpecialEntities1(input string) string {
	var output string
	output = strings.Replace(input, "&amp;", "&", -1)
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
	var b strings.Builder
	b.Grow(len(input))
	i := 0
	for i < len(input) {
		if input[i] == '&' {
			switch {
			case strings.HasPrefix(input[i:], "&amp;"):
				b.WriteString("&")
				i += len("&amp;")
			case strings.HasPrefix(input[i:], "&lt;"):
				b.WriteString("<")
				i += len("&lt;")
			case strings.HasPrefix(input[i:], "&gt;"):
				b.WriteString(">")
				i += len("&gt;")
			case strings.HasPrefix(input[i:], "&quot;"):
				b.WriteString("\"")
				i += len("&quot;")
			case strings.HasPrefix(input[i:], "&lsquo;"):
				b.WriteString("‘")
				i += len("&lsquo;")
			case strings.HasPrefix(input[i:], "&rsquo;"):
				b.WriteString("’")
				i += len("&rsquo;")
			case strings.HasPrefix(input[i:], "&tilde;"):
				b.WriteString("~")
				i += len("&tilde;")
			case strings.HasPrefix(input[i:], "&ndash;"):
				b.WriteString("–")
				i += len("&ndash;")
			case strings.HasPrefix(input[i:], "&mdash;"):
				b.WriteString("—")
				i += len("&mdash;")
			case strings.HasPrefix(input[i:], "&apos;"):
				b.WriteString("'")
				i += len("&apos;")
			default:
				b.WriteByte(input[i])
				i++
			}
		} else {
			b.WriteByte(input[i])
			i++
		}
	}
	return b.String()
}

func main() {
	input := "&amp;&lt;&gt;&quot;&lsquo;&rsquo;&tilde;&ndash;&mdash;&apos;"
	fmt.Println(ReplaceHTMLSpecialEntities1(input))
	fmt.Println(ReplaceHTMLSpecialEntities2(input))
}

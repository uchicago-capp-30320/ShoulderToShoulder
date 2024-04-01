import { Component } from '@angular/core';
import { Input } from '@angular/core';

@Component({
  selector: 'app-header',
  templateUrl: './app-header.component.html',
  styleUrl: './app-header.component.css'
})
export class AppHeaderComponent {
  @Input() title: string = "Welcome to Shoulder to Shoulder";
  @Input() subtitle: string = "You can use this form to generate an individual event. To connect with community members, please create an account.";

  constructor() { }

  ngOnInit() {
  }
}
